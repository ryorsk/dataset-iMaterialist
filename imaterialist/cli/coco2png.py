import json
from pathlib import Path

import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image

from pycocotools.coco import COCO

from segmentation.transforms import segmap_to_pil

PROJECT_ROOT = Path(__file__).parents[2]


def main():
    coco2png()


def coco2png():
    df_ctg = get_categories()

    ano_json: Path = PROJECT_ROOT / "raw/instances_attributes_val2020.json"
    ann_out_dir: Path = PROJECT_ROOT / "raw/ann"
    map_out_dir: Path = PROJECT_ROOT / "raw/map"
    coco = COCO(ano_json)

    imgIds = coco.getImgIds()
    print(len(imgIds))

    ann_out_dir.mkdir(parents=True, exist_ok=True)
    map_out_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(0, len(imgIds))):
        img = coco.loadImgs(imgIds)[i]

        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(
            imgIds=img["id"], catIds=cat_ids, iscrowd=None
        )
        anns = coco.loadAnns(anns_ids)
        anns_img = np.zeros((img["height"], img["width"]))
        image_id = img["kaggle_id"]

        for ann in anns:
            # Ignore classes not included in iMaterialist
            if ann["category_id"] > 26:
                continue
            anns_img = np.maximum(
                anns_img, coco.annToMask(ann) * (ann["category_id"] + 1)
            )

        ann_img = segmap_to_pil(anns_img, df_ctg.index)
        map_img = Image.fromarray(anns_img).convert("L")
        ann_img.save(ann_out_dir / (image_id + ".png"))
        map_img.save(map_out_dir / (image_id + ".png"))


def get_categories():
    meta_json: Path = PROJECT_ROOT / "raw/label_descriptions.json"

    with open(meta_json) as f:
        meta = json.load(f)
    df_ctg = pd.DataFrame(meta["categories"])
    # Shift category_id by 1 bacause background's id is 0
    df_ctg["id"] += 1
    # Remove some categories
    df_ctg = df_ctg[
        ~df_ctg["supercategory"].isin(
            ["garment parts", "closures", "decorations"]
        )
    ]
    df_ctg = df_ctg.set_index("id")
    # Append `background` to the last row
    df_ctg.loc[0] = {
        "name": "background",
        "supercategory": "background",
        "level": 2,
    }
    # Move `background` to the first row
    df_ctg = df_ctg.sort_index()

    return df_ctg


if __name__ == "__main__":
    main()
