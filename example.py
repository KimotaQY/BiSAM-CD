import argparse
import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from inference import inference


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, help="SAM2's checkpoint directory")
    parser.add_argument(
        "--config_dir",
        type=str,
        help="SAM2's config directory. Usually /Your project/sam2/sam2/configs/sam2.1, it is recommended to use an absolute path.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="b+",
        help="parameter for sam2, include t, s, b+, l",
    )
    parser.add_argument("--mid_frame", type=int, default=1, help="number of mid frames")
    parser.add_argument("--diff_frame_num", type=int, default=-1)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--label_origin", type=str, default="whu")
    parser.add_argument("--prompt_type", type=str, default="box")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    print(args)

    label_origin = args.label_origin
    if label_origin == "whu":
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
        mask = inference(
            img_paths=img_paths,
            label_paths=label_paths,
            **vars(args),
        )
    else:
        # label from owlv2
        img_name = "tile_13312_26624.png"
        img_dirs = [
            "example/WHU-CD/test/A",
            "example/WHU-CD/test/B",
        ]
        label_dirs = [
            "example/WHU-CD/test/A_owlv2_large_[single building]",
            "example/WHU-CD/test/B_owlv2_large_[single building]",
        ]

        img_paths = []
        for img_dir in img_dirs:
            img_paths.append(os.path.join(img_dir, img_name))
        label_paths = []
        for label_dir in label_dirs:
            label_paths.append(os.path.join(label_dir, Path(img_name).stem + ".json"))
        mask = inference(
            img_paths=img_paths,
            label_paths=label_paths,
            **vars(args),
        )

    # create a figure that can hold three subplots
    plt.figure(figsize=(15, 5))  # set the figure size

    # drawing img_A
    img_A = cv2.imread(img_paths[0])
    plt.subplot(1, 3, 1)
    plt.imshow(img_A)
    plt.title("T1")
    plt.axis("off")

    # drawing img_B
    img_B = cv2.imread(img_paths[1])
    plt.subplot(1, 3, 2)
    plt.imshow(img_B)
    plt.title("T2")
    plt.axis("off")

    # drawing mask
    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap="gray")
    plt.title("mask")
    plt.axis("off")

    # show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
