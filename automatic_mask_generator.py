import json
import os

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

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


np.random.seed(3)


def show_anns(anns, borders=True, output_path=None):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.3]])
        img[m] = color_mask
        if borders:
            import cv2

            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # Try to smooth contours
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.5), thickness=1)

            if output_path is not None:
                cv2.imwrite(output_path, img * 255)
    ax.imshow(img)


def overlay_mask_on_image_and_save(image, anns, output_path=None, borders=True):
    if len(anns) == 0:
        return

    sorted_anns = sorted(anns, key=lambda x: x["area"], reverse=True)
    img = np.zeros_like(image)

    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate(
            [np.random.random(3), [0.5]]
        )  # Random color with alpha=0.5
        img[m] = color_mask[:3] * 255  # Apply RGB values without alpha channel

        if borders:
            import cv2

            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # Try to smooth contours
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            cv2.drawContours(img, contours, -1, (0, 0, 255), thickness=1)

    # Overlay the mask on the original image
    overlayed_img = cv2.addWeighted(image, 1, img, 0.5, 0)

    # Save the result
    if output_path is not None:
        cv2.imwrite(output_path, overlayed_img)


def save_mask_as_json(masks, output_dir, image_name):
    # 创建输出目录
    # output_dir = "output_masks_coco"
    os.makedirs(output_dir, exist_ok=True)

    # 准备 JSON 数据
    json_data = []

    for i, mask_info in enumerate(masks):
        # 提取 RLE 编码
        rle = mask_info["segmentation"]

        # 将 RLE 转换为二值掩码（可选：用于调试或可视化）
        # binary_mask = mask_utils.decode(rle)

        # 构造要保存的数据结构
        mask_entry = {
            "id": i,
            "area": int(mask_info["area"]),
            "bbox": [int(x) for x in mask_info["bbox"]],  # 转换为整数列表
            "point_coords": mask_info["point_coords"],
            "predicted_iou": float(mask_info["predicted_iou"]),
            "stability_score": float(mask_info["stability_score"]),
            "rle": {"size": rle["size"], "counts": rle["counts"]},
        }

        json_data.append(mask_entry)

    # 获取文件名除了后缀
    image_name = os.path.splitext(image_name)[0]
    # 保存为 JSON 文件
    output_json_path = os.path.join(output_dir, f"{image_name}.json")
    with open(output_json_path, "w") as f:
        json.dump(json_data, f, indent=4)

    print(f"掩码已保存为 COCO RLE 格式的 JSON 文件：{output_json_path}")


from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import gc

if __name__ == "__main__":
    # 加载SAM2
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
    # dataset_name = "SECOND"
    # for model_type in ["l"]:
    #     for label_type in ["im2_1024", "im1_1024"]:
    #         # model_type = "t"
    #         checkpoint = model_obj[model_type]["checkpoint"]
    #         config = model_obj[model_type]["config"]
    #         sam2_checkpoint = os.path.join("E:/CD_Checkpoints", checkpoint)
    #         model_cfg = os.path.join(
    #             "E:/CD_projects/sam2-cd-no-training/sam2/configs/sam2.1", config
    #         )

    #         sam2 = build_sam2(
    #             model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False
    #         )

    #         mask_generator = SAM2AutomaticMaskGenerator(
    #             model=sam2,
    #             points_per_side=64,
    #             points_per_batch=128,
    #             pred_iou_thresh=0.7,
    #             stability_score_thresh=0.92,
    #             stability_score_offset=0.7,
    #             crop_n_layers=1,
    #             box_nms_thresh=0.7,
    #             crop_n_points_downscale_factor=2,
    #             min_mask_region_area=25.0,
    #             output_mode="coco_rle",
    #             use_m2m=True,
    #         )

    #         # 创建输出目录
    #         output_dir = f"E:/CD_datasets/{dataset_name}/test/sam2/{label_type}_sam2_coco_rle_{model_type}"
    #         # 判断输出目录是否存在
    #         os.makedirs(output_dir, exist_ok=True)
    #         json_result_names = os.listdir(output_dir)
    #         json_result_names = [
    #             os.path.splitext(json_result_name)[0]
    #             for json_result_name in json_result_names
    #         ]
    #         # 读取图片名，过滤非文件名后缀
    #         img_dir = f"E:/CD_datasets/{dataset_name}/test/{label_type}"
    #         img_names = os.listdir(img_dir)
    #         img_names = [
    #             img_name for img_name in img_names if img_name.endswith(".png")
    #         ]
    #         for i, img_name in enumerate(img_names):
    #             # 跳过已存在的文件
    #             if os.path.splitext(img_name)[0] in json_result_names:
    #                 print(f"Skipping image {i+1}/{len(img_names)}: {img_name}")
    #                 continue
    #             else:
    #                 print(f"Processing image {i+1}/{len(img_names)}: {img_name}")

    #             image = Image.open(os.path.join(img_dir, img_name))
    #             image = np.array(image.convert("RGB"))

    #             masks = mask_generator.generate(image)

    #             # 保存掩码为JSON文件
    #             save_mask_as_json(masks, output_dir, img_name)

    #             # overlay_mask_on_image_and_save(image, masks)

    #             # plt.figure(figsize=(20, 20))
    #             # plt.imshow(image)
    #             # show_anns(masks)
    #             # plt.axis("off")
    #             # plt.show()

    #             # 清理显存和内存
    #             del image, masks
    #             torch.cuda.empty_cache()
    #             gc.collect()

    # 单张图片测试
    model_type = "l"
    checkpoint = model_obj[model_type]["checkpoint"]
    config = model_obj[model_type]["config"]
    sam2_checkpoint = os.path.join("E:/CD_Checkpoints", checkpoint)
    model_cfg = os.path.join(
        "E:/CD_projects/sam2-cd-no-training/sam2/configs/sam2.1", config
    )

    sam2 = build_sam2(
        model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False
    )

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=64,
        points_per_batch=128,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        stability_score_offset=0.7,
        crop_n_layers=1,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=25.0,
        # output_mode="coco_rle",
        use_m2m=True,
    )

    for i_folder in ["A", "B"]:
        img_dir = f"E:/CD_datasets/LEVIR-CD/test/{i_folder}"
        img_name = "test_59.png"
        image = Image.open(os.path.join(img_dir, img_name))
        image = np.array(image.convert("RGB"))

        masks = mask_generator.generate(image)

        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        # show_anns(masks, output_path="./output_jpg/B_test_84.png")
        overlay_mask_on_image_and_save(
            image, masks, output_path=f"./output_jpg/{i_folder}_{img_name}.png"
        )
        plt.axis("off")
        # plt.show()
        plt.close()
