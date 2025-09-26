import os
from pathlib import Path
import cv2
import torch
import matplotlib.pyplot as plt

from BiSAM_CD import step_one
from sam2.build_sam import build_sam2_video_predictor
from utils.get_annos import get_annos
from eval import sum_masks_dict

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


def main(
    img_paths: list,
    label_paths: list,
    model_type="b+",
    mid_frame=0,
    diff_frame_num=-1,
    iou_threshold=0.5,
    label_origin="whu",
    prompt_type="box",
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
    # 加载SAM2 video predictor
    sam2_checkpoint = os.path.join("E:/CD_Checkpoints", checkpoint)
    model_cfg = os.path.join("E:/CD_projects/BiSAM-CD/sam2/configs/sam2.1", config)

    if None in [img_dirs, label_paths]:
        print("请输入前后时相图片路径和标签路径")
        return

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    # 获取标签
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

    # 创建一个绘图窗口并绘制三个子图
    plt.figure(figsize=(15, 5))  # 设置窗口大小

    # 绘制 img_A
    img_A = cv2.imread(img_paths[0])
    plt.subplot(1, 3, 1)
    plt.imshow(img_A)
    plt.title("T1")
    plt.axis("off")

    # 绘制 img_B
    img_B = cv2.imread(img_paths[1])
    plt.subplot(1, 3, 2)
    plt.imshow(img_B)
    plt.title("T2")
    plt.axis("off")

    # 绘制 mask
    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap="gray")
    plt.title("mask")
    plt.axis("off")

    # 展示图像
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # label from whu
    img_name = "tile_13312_26624.png"
    img_dirs = [
        "example/WHU-CD/test/A",
        "example/WHU-CD/test/B",
    ]
    label_dirs = [
        "example/WHU-CD/before_label",
        "example/WHU-CD/after_label",
    ]
    img_paths = []
    for img_dir in img_dirs:
        img_paths.append(os.path.join(img_dir, img_name))
    label_paths = []
    for label_dir in label_dirs:
        label_paths.append(os.path.join(label_dir, img_name))
    main(
        img_paths=img_paths,
        label_paths=label_paths,
        label_origin="whu",
        prompt_type="box",
        mid_frame=1,
    )

    # # label from owlv2
    # img_name = "tile_13312_26624.png"
    # img_dirs = [
    #     "example/WHU-CD/test/A",
    #     "example/WHU-CD/test/B",
    # ]
    # label_dirs = [
    #     "example/WHU-CD/test/A_owlv2_large_[single building]",
    #     "example/WHU-CD/test/B_owlv2_large_[single building]",
    # ]

    # img_paths = []
    # for img_dir in img_dirs:
    #     img_paths.append(os.path.join(img_dir, img_name))
    # label_paths = []
    # for label_dir in label_dirs:
    #     label_paths.append(os.path.join(label_dir, Path(img_name).stem + ".json"))
    # main(
    #     img_paths=img_paths,
    #     label_paths=label_paths,
    #     label_origin="whu_owlv2",
    #     prompt_type="box",
    #     mid_frame=1,
    # )
