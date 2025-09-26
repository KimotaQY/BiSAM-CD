import json

import numpy as np

from utils.extract_masks import extract_masks
from pycocotools import mask as mask_utils


def get_annos(label_path, label_origin):
    if label_origin != "whu":
        with open(label_path, "r") as f:
            json_result = json.load(f)
            annos = json_result if label_origin == "sam2" else json_result["objects"]
    else:
        # 获取label中建筑物mask、box、points，并逐个赋予id进行追踪
        annos = extract_masks(label_path)
    print(f"建筑物数量: {len(annos)}")

    # 格式化annos:
    format_annos = []
    for idx, item in enumerate(annos):
        anno = {}
        if label_origin == "whu":
            mask, (x, y, w, h), points = item.values()
            anno = {
                "points": points,
                "box": np.array([x, y, x + w, y + h]),
                "mask": mask,
            }
        else:
            box = item.get("bbox")
            anno = {"box": np.array(box)}
            if label_origin == "sam2":
                rle = item.get("rle")
                area = item.get("area")
                # 忽略面积小于150的mask，根据需求调整
                if area < 150:
                    continue
                anno["mask"] = mask_utils.decode(rle) * 255

        format_annos.append(anno)

    return format_annos
