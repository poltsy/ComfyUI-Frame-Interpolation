import torch
import pathlib
from fractions import Fraction
from vfi_utils import load_file_from_github_release, preprocess_frames, postprocess_frames, InterpolationStateList
import typing
from comfy.model_management import get_torch_device, soft_empty_cache
from packaging import version
from comfy.utils import ProgressBar

MODEL_TYPE = pathlib.Path(__file__).parent.name
CKPT_NAME_VER_DICT = {
    "rife47.pth": "4.7",
    "rife49.pth": "4.7",
    "rife417.pth": "4.17",
    "rife426.pth": "4.26",
    "sudo_rife4_269.662_testV1_scale1.pth": "4.0",
    # Arch 4.10 doesn't work due to state dict mismatch
    # "rife410.pth": "4.10",
    # "rife411.pth": "4.10",
    # "rife412.pth": "4.10"
}

DTYPE_OPTIONS = ["float32", "float16", "bfloat16"]
DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Module-level model cache: avoids reloading weights on every node execution.
# Key: (ckpt_name, dtype, torch_compile) — invalidated if any of these change.
_model_cache: typing.Dict[typing.Tuple, torch.nn.Module] = {}

def _valid_dtype(precision: str, device):
    if device.type != "cuda":
        print(f"[RIFE VFI] non-CUDA device ({device}); forcing fp32.")
        return torch.float32
    if precision == "bfloat16":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    return DTYPE_MAP.get(precision, torch.float32)

class RIFE_VFI:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (
                    sorted(list(CKPT_NAME_VER_DICT.keys()), key=lambda ckpt_name: version.parse(CKPT_NAME_VER_DICT[ckpt_name])),
                    {"default": "rife426.pth"}
                ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 1000, "min": 1, "max": 10000}),
                "fps_mode": ("BOOLEAN", {"default": False,
                                         "tooltip": "When enabled, use source_fps and target_fps to determine output frame count "
                                                    "instead of multiplier."}),
                "multiplier": ("INT", {"default": 2, "min": 1,
                                       "tooltip": "Used when fps_mode is off. Multiplies each input frame pair by this factor."}),
                "fast_mode": ("BOOLEAN", {"default": True}),
                "ensemble": ("BOOLEAN", {"default": True}),
                "scale_factor": ([0.25, 0.5, 1.0, 2.0, 4.0], {"default": 1.0}),
                "dtype": (DTYPE_OPTIONS, {"default": "bfloat16"}),
                "torch_compile": ("BOOLEAN", {"default": False,
                                              "tooltip": "Compile the model with torch.compile() for 10-30% faster inference "
                                                         "after the first (warm-up) run. Requires PyTorch 2.0+."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 1, "tooltip": "Does nothing. "}),
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES", ),
                "source_fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.001,
                                         "tooltip": "Frame rate of the input frames. "
                                                    "Set both source_fps and target_fps (>0) to use FPS mode instead of multiplier."}),
                "target_fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.001,
                                         "tooltip": "Desired output frame rate. "
                                                    "Set both source_fps and target_fps (>0) to use FPS mode instead of multiplier."}),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"

    def vfi(
        self,
        ckpt_name: typing.AnyStr,
        frames: torch.Tensor,
        clear_cache_after_n_frames: int = 1000,
        fps_mode: bool = False,
        multiplier: typing.SupportsInt = 2,
        fast_mode: bool = True,
        ensemble: bool = True,
        scale_factor: float = 1.0,
        dtype: str = "bfloat16",
        torch_compile: bool = False,
        batch_size: int = 1,
        optional_interpolation_states: InterpolationStateList = None,
        source_fps: float = 0.0,
        target_fps: float = 0.0,
        **kwargs
    ):
        """
        Perform video frame interpolation using a given checkpoint model.

        Args:
            ckpt_name (str): The name of the checkpoint model to use.
            frames (torch.Tensor): A tensor containing input video frames.
            clear_cache_after_n_frames (int, optional): The number of frames to process before clearing CUDA cache
                to prevent memory overflow. Defaults to 10. Lower numbers are safer but mean more processing time.
                How high you should set it depends on how many input frames there are, input resolution (after upscaling),
                how many times you want to multiply them, and how long you're willing to wait for the process to complete.
            multiplier (int, optional): The multiplier for each input frame. 60 input frames * 2 = 120 output frames. Defaults to 2.
            dtype (str, optional): Floating-point precision for inference.
                "float32" — full precision, safest (default).
                "float16" — half precision, ~2x faster and ~50% less VRAM, may have minor artifacts.
                "bfloat16" — brain float16, better numeric range than float16, requires Ampere+ GPU (RTX 30xx+).
            torch_compile (bool, optional): Compile the model with torch.compile() for 10-30% speedup.
                The first inference call will be slow (compilation warm-up); subsequent calls are faster.
                Requires PyTorch 2.0+. Defaults to False.
            batch_size (int, optional): Number of interpolation tasks per GPU call. Each task produces one
                intermediate frame. Higher values improve throughput but use more VRAM. Defaults to 1.

        Returns:
            tuple: A tuple containing the output interpolated frames.
        """
        from .rife_arch import IFNet

        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        arch_ver = CKPT_NAME_VER_DICT[ckpt_name]
        device = get_torch_device()
        torch_dtype = _valid_dtype(dtype, device)

        #  4.26 disable ensemble
        if arch_ver == "4.26":
            ensemble = False

        # Cache the model by (ckpt_name, dtype, torch_compile) so repeated node
        # executions skip the load_state_dict + device transfer entirely.
        cache_key = (ckpt_name, dtype, torch_compile)
        if cache_key not in _model_cache:
            interpolation_model = IFNet(arch_ver=arch_ver)

            # 加载状态字典
            state_dict = torch.load(model_path, weights_only=False)

            # rife426.pth contains teacher and caltime prefixes that need filtering
            if ckpt_name == "rife426.pth":
                filtered_state_dict = {}
                for k, v in state_dict.items():
                    if not k.startswith('teacher.') and not k.startswith('caltime.'):
                        filtered_state_dict[k] = v
                interpolation_model.load_state_dict(filtered_state_dict, strict=False)
            else:
                interpolation_model.load_state_dict(state_dict)

            if torch_dtype != torch.float32:
                interpolation_model = interpolation_model.to(torch_dtype)
            interpolation_model.eval().to(device)
            del state_dict
            soft_empty_cache()

            if torch_compile:
                interpolation_model = torch.compile(interpolation_model)
            _model_cache[cache_key] = interpolation_model
            print(f"[RIFE VFI] Loaded and cached model {ckpt_name} ({dtype}{'+ torch.compile' if torch_compile else ''})", end=' ')
        else:
            interpolation_model = _model_cache[cache_key]
            print(f"[RIFE VFI] Using cached model {ckpt_name} ({dtype}{'+ torch.compile' if torch_compile else ''})", end =' ')

        frames_nchw = preprocess_frames(frames)  # NCHW float32 on CPU

        if arch_ver == "4.26":
            scale_list = [16 / scale_factor, 8 / scale_factor, 4 / scale_factor, 2 / scale_factor, 1 / scale_factor]
        else:
            scale_list = [8 / scale_factor, 4 / scale_factor, 2 / scale_factor, 1 / scale_factor]

        n_input = len(frames_nchw)
        n_pairs = n_input - 1
        use_fps_mode = fps_mode and source_fps > 0 and target_fps > 0

        # Compute total_output first, then allocate a single NHWC fp32 tensor.
        h, w = frames_nchw.shape[2], frames_nchw.shape[3]
        c = frames_nchw.shape[1]
        output_frames: torch.Tensor  # allocated below
        tasks: typing.List[typing.Tuple[int, float]] = []
        output_specs: typing.List[typing.Tuple] = []

        if use_fps_mode:
            # Convert source/target fps to Fractions
            src_frac = Fraction(str(source_fps)).limit_denominator(1000)
            tgt_frac = Fraction(str(target_fps)).limit_denominator(1000)
            if abs(float(src_frac - tgt_frac)) < 0.01:
                print("source_fps ≈ target_fps, returning frames unchanged.")
                return (postprocess_frames(frames_nchw.to(torch.float32)),)

            # Estimate output count for the iteration range.
            n_output = max(2, int((n_input - 1) * tgt_frac / src_frac) + 1)

            # Build a schedule: for each output tick compute the source pair
            # index and interpolation alpha using exact Fraction arithmetic.
            for out_i in range(n_output):
                # Exact continuous position in input-frame space (Fraction).
                num = Fraction(out_i) * (n_input - 1) / (n_output - 1)
                pair_idx = int(num)
                alpha = float(num - pair_idx)

                if pair_idx >= n_pairs or alpha < 1e-6 or alpha > 1 - 1e-6:
                    output_specs.append(('orig', min(pair_idx + int(alpha > 0.5), n_input - 1)))
                elif optional_interpolation_states is not None and optional_interpolation_states.is_frame_skipped(pair_idx):
                    output_specs.append(('orig', pair_idx))
                else:
                    output_specs.append(('interp', len(tasks)))
                    tasks.append((pair_idx, alpha))
            print(f"FPS mode {source_fps} → {target_fps} fps  ({n_input} → {n_output}, {len(tasks)} generated frames)")

        else: # Multiplier mode
            if isinstance(multiplier, int):
                multipliers = [int(multiplier)] * n_pairs
            else:
                multipliers = list(map(int, multiplier))
                multipliers += [2] * (n_pairs - len(multipliers))

            # Precompute total output count.
            tasks_remaining_per_pair: typing.Dict[int, int] = {}
            n_output = 0
            for pair_idx in range(n_pairs):
                n_output += 1  # original frame at this position
                if optional_interpolation_states is not None and optional_interpolation_states.is_frame_skipped(pair_idx):
                    tasks_remaining_per_pair[pair_idx] = 0
                    continue
                m = multipliers[pair_idx]
                n_steps = max(m - 1, 0)
                tasks_remaining_per_pair[pair_idx] = n_steps
                n_output += n_steps
            n_output += 1  # trailing original frame

            for pair_idx in range(n_pairs):
                output_specs.append(('orig', pair_idx))
                if optional_interpolation_states is not None and optional_interpolation_states.is_frame_skipped(pair_idx):
                    continue
                m = multipliers[pair_idx]
                for step in range(1, m):
                    output_specs.append(('interp', len(tasks)))
                    tasks.append((pair_idx, step / m))
            output_specs.append(('orig', n_input - 1))
            print(f"Multiplier mode {multiplier}x fps ({len(frames_nchw)} → {n_output}, {len(tasks)} generated frames)")

        # output_specs interleaves ('orig', src_idx) and ('interp', task_pos)
        # entries. Walk it once: for every 'interp' entry record which slot
        # that task position writes into.
        task_slot_map = [0] * len(tasks)  # task_pos → output slot index
        out_slot = 0
        for spec in output_specs:
            if spec[0] == 'orig':
                out_slot += 1
            else:
                task_slot_map[spec[1]] = out_slot
                out_slot += 1

        n_output_actual = len(output_specs)
        output_frames = torch.empty(n_output_actual, h, w, c, dtype=torch.float32)

        # Fill original frames into their exact slots up front.
        if use_fps_mode:
            out_slot = 0
            for spec in output_specs:
                if spec[0] == 'orig':
                    src_idx = spec[1]
                    output_frames[out_slot] = frames_nchw[src_idx].permute(1, 2, 0)
                out_slot += 1
        else:
            fill_pos = 0
            for pair_idx in range(n_pairs):
                output_frames[fill_pos] = frames_nchw[pair_idx].permute(1, 2, 0)
                n_interp = tasks_remaining_per_pair.get(pair_idx, 0)
                fill_pos += 1 + n_interp
            output_frames[fill_pos] = frames_nchw[-1].permute(1, 2, 0)

        staging = None
        if device.type == "cuda":
            staging = torch.empty(c, h, w, dtype=torch_dtype, pin_memory=True)

        comfy_pbar = ProgressBar(len(tasks))
        frames_processed_since_cache_clear = 0
        pos = 0

        # reuse frame1 of pair N as frame0 of pair N+1
        gpu_frame1 = None       # last uploaded frame1 tensor on GPU
        gpu_pair_idx = -1       # which pair its frame1 belongs to

        with torch.inference_mode():
            while pos < len(tasks):
                pair_idx, alpha = tasks[pos]

                # reuse the already-uploaded frame0.
                if gpu_frame1 is not None and pair_idx == gpu_pair_idx + 1:
                    frame0_batch = gpu_frame1
                else:
                    frame0_batch = frames_nchw[pair_idx : pair_idx + 1].to(device, dtype=torch_dtype)

                frame1_batch = frames_nchw[pair_idx + 1 : pair_idx + 2].to(device, dtype=torch_dtype)
                timestep_tensor = torch.tensor([alpha],
                    dtype=torch_dtype, device=device).view(1, 1, 1, 1)
                middle_frames = interpolation_model(
                    frame0_batch,
                    frame1_batch,
                    timestep_tensor,
                    scale_list,
                    fast_mode,
                    ensemble,
                ).clamp(0, 1).detach()
                write_slot = task_slot_map[pos]

                if staging is not None:
                    staging.copy_(middle_frames[0], non_blocking=True)
                    torch.cuda.current_stream(device).synchronize()
                    output_frames[write_slot].copy_(staging.permute(1, 2, 0))
                else:
                    output_frames[write_slot].copy_(
                        middle_frames[0]
                        .permute(1, 2, 0)
                        .to(dtype=torch.float32, device="cpu")
                    )

                cur_frame = pos + 1
                print(f"[RIFE VFI] Generating frame {cur_frame} / {len(tasks)}  ", end='\r')
                comfy_pbar.update_absolute(cur_frame, len(tasks))
                frames_processed_since_cache_clear += 1

                if not use_fps_mode:
                    tasks_remaining_per_pair[pair_idx] -= 1
                    if tasks_remaining_per_pair[pair_idx] == 0:
                        frames_processed_since_cache_clear += 1

                if frames_processed_since_cache_clear >= clear_cache_after_n_frames:
                    soft_empty_cache()
                    frames_processed_since_cache_clear = 0

                # Update frame reuse state.
                gpu_frame1 = frame1_batch.detach().clone()
                gpu_pair_idx = pair_idx
                pos += 1

        del interpolation_model
        soft_empty_cache()
        ready_frames = output_frames[:n_output_actual].to(torch.float32)
        del output_frames
        soft_empty_cache()
        return (postprocess_frames(ready_frames.permute(0, 3, 1, 2)),)
