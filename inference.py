import gc
import os
import torch

from BiSAM_CD import step_one
from sam2.build_sam import build_sam2_video_predictor
from utils.get_annos import get_annos
from utils.sum_masks_dict import sum_masks_dict

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


def inference(
    img_paths: list,
    label_paths: list,
    model_type="b+",
    mid_frame=0,
    diff_frame_num=-1,
    iou_threshold=0.5,
    label_origin="whu",
    prompt_type="box",
    **kwargs,
):
    model_obj = {
        "t": {
            "checkpoint": "sam2.1_hiera_tiny.pt",
            "config": "sam2.1_hiera_t.yaml",
        },
        "s": {
            "checkpoint": "sam2.1_hiera_small.pt",
            "config": "sam2.1_hiera_s.yaml",
        },
        "b+": {
            "checkpoint": "sam2.1_hiera_base_plus.pt",
            "config": "sam2.1_hiera_b+.yaml",
        },
        "l": {
            "checkpoint": "sam2.1_hiera_large.pt",
            "config": "sam2.1_hiera_l.yaml",
        },
    }
    checkpoint = model_obj[model_type]["checkpoint"]
    config = model_obj[model_type]["config"]
    # load SAM2 video predictor
    ckpt_dir = kwargs.get("ckpt_dir", "")
    config_dir = kwargs.get("config_dir", "")
    sam2_checkpoint = os.path.join(ckpt_dir, checkpoint)
    model_cfg = os.path.join(config_dir, config)

    if None in [img_paths, label_paths]:
        print("Please input img_paths and label_paths")
        return

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    # get annotations
    annos_list = []
    for label_path in label_paths:
        annos_list.append(get_annos(label_path, label_origin))

    diff_mask_list = step_one(
        img_paths,
        annos_list,
        predictor,
        mid_frame=mid_frame,
        diff_frame_num=diff_frame_num,
        iou_threshold=iou_threshold,
        prompt_type=prompt_type,
        # prompts=prompts,
    )

    diff_mask = sum_masks_dict(*diff_mask_list, iou_threshold=iou_threshold)

    h, w = diff_mask.shape[-2:]
    mask = diff_mask.reshape(h, w, 1)

    if "predictor" in locals():
        del predictor
    torch.cuda.empty_cache()
    gc.collect()

    return mask
