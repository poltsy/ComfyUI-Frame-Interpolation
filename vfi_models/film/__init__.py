import torch
from comfy.model_management import get_torch_device, soft_empty_cache
import bisect
import numpy as np
import typing
from vfi_utils import InterpolationStateList, load_file_from_github_release, preprocess_frames, postprocess_frames
import pathlib
import gc
import comfy.utils
import sys
import warnings

# Suppress PyTorch UserWarning about padding with even kernel lengths
warnings.filterwarnings("ignore", message="Using padding='same' with even kernel lengths and odd dilation")

class VFIProgressBar:
    """A progress bar that displays both in ComfyUI UI and terminal"""
    def __init__(self, total, desc="FILM VFI"):
        self.total = total
        self.n = 0
        self.desc = desc
        self.comfy_pbar = comfy.utils.ProgressBar(total)
        self._print_terminal()

    def update(self, n=1):
        self.n += n
        self.comfy_pbar.update(n)
        self._print_terminal()

    def _print_terminal(self):
        if self.total > 0:
            percent = 100 * (self.n / float(self.total))
            bar_length = 40
            filled_length = int(bar_length * self.n // self.total)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            sys.stdout.write(f'\r{self.desc}: [{bar}] {percent:.1f}%')
            sys.stdout.flush()
            if self.n >= self.total:
                sys.stdout.write('\n')
                sys.stdout.flush()

MODEL_TYPE = pathlib.Path(__file__).parent.name
DEVICE = get_torch_device()
MODEL_CACHE = {}

@torch.inference_mode()
def inference(model, img_batch_1, img_batch_2, inter_frames, model_dtype):
    results = [
        img_batch_1,
        img_batch_2
    ]

    idxes = [0, inter_frames + 1]
    remains = list(range(1, inter_frames + 1))

    splits = torch.linspace(0, 1, inter_frames + 2)

    for _ in range(len(remains)):
        starts = splits[idxes[:-1]]
        ends = splits[idxes[1:]]
        distances = ((splits[None, remains] - starts[:, None]) / (ends[:, None] - starts[:, None]) - .5).abs()
        matrix = torch.argmin(distances).item()
        start_i, step = np.unravel_index(matrix, distances.shape)
        end_i = start_i + 1

        x0 = results[start_i]
        x1 = results[end_i]

        # dt calculation
        dt_val = (splits[remains[step]] - splits[idxes[start_i]]) / (splits[idxes[end_i]] - splits[idxes[start_i]])
        dt = torch.tensor([[dt_val]], device=DEVICE, dtype=model_dtype)

        prediction = model(x0, x1, dt)

        insert_position = bisect.bisect_left(idxes, remains[step])
        idxes.insert(insert_position, remains[step])
        results.insert(insert_position, prediction.clamp(0, 1))
        del remains[step]

    return results

class FILM_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (["film_net_fp32.pt", "film_net_fp16.pt"], ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 1000}),
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES", )
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"

    @torch.inference_mode()
    def vfi(
        self,
        ckpt_name: typing.AnyStr,
        frames: torch.Tensor,
        clear_cache_after_n_frames = 10,
        multiplier: typing.SupportsInt = 2,
        optional_interpolation_states: InterpolationStateList = None,
        **kwargs
    ):
        interpolation_states = optional_interpolation_states

        if ckpt_name not in MODEL_CACHE:
            model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
            model = torch.jit.load(model_path, map_location='cpu')
            model.eval()
            model = model.to(DEVICE)

            # Determine model dtype
            try:
                model_dtype = next(model.parameters()).dtype
            except StopIteration:
                model_dtype = torch.float16 if "fp16" in ckpt_name else torch.float32

            MODEL_CACHE[ckpt_name] = (model, model_dtype)

        model, model_dtype = MODEL_CACHE[ckpt_name]
        dtype = torch.float32

        frames = preprocess_frames(frames).pin_memory()
        number_of_frames_processed_since_last_cleared_cuda_cache = 0

        if isinstance(multiplier, int):
            multipliers = [multiplier] * (len(frames) - 1)
        else:
            multipliers = list(map(int, multiplier))
            multipliers += [2] * (len(frames) - len(multipliers) - 1)

        # Pre-allocate output list for better memory management
        total_output_frames = sum(multipliers) + 1
        output_frames = [None] * total_output_frames
        output_index = 0


        # Initialize progress bar (both UI and terminal)
        total_pairs = len(frames) - 1
        pbar = VFIProgressBar(total_pairs, desc="FILM VFI")

        for frame_itr in range(len(frames) - 1):
            if interpolation_states is not None and interpolation_states.is_frame_skipped(frame_itr):
                continue

            # Ensure that input frames are in the same dtype as model
            frame_0 = frames[frame_itr:frame_itr+1].to(DEVICE, non_blocking=True).to(model_dtype)
            frame_1 = frames[frame_itr+1:frame_itr+2].to(DEVICE, non_blocking=True).to(model_dtype)

            # Use the recursive inference which is better for FILM's motion estimation
            results = inference(model, frame_0, frame_1, multipliers[frame_itr] - 1, model_dtype)

            # Move results to CPU immediately to free GPU memory
            for f in results[:-1]:
                output_frames[output_index] = f.detach().to(device="cpu", dtype=dtype, non_blocking=True)
                output_index += 1

            number_of_frames_processed_since_last_cleared_cuda_cache += 1
            if number_of_frames_processed_since_last_cleared_cuda_cache >= clear_cache_after_n_frames:
                soft_empty_cache()
                number_of_frames_processed_since_last_cleared_cuda_cache = 0

            # Update progress bar (both UI and terminal)
            pbar.update(1)

        output_frames[output_index] = frames[-1:].to(dtype=dtype) # Append final frame

        # Filter out None values in case of skipped frames
        output_frames = [f for f in output_frames if f is not None]

        out = torch.cat(output_frames, dim=0)

        soft_empty_cache()
        return (postprocess_frames(out), )


NODE_CLASS_MAPPINGS = {
    "FILM VFI": FILM_VFI,
}