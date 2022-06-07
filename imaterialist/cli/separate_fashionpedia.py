import os
import shutil
from tqdm import tqdm
from pycocotools.coco import COCO

DATASET_ROOT = "/Users/ryosuke.araki/dataset/Fashionpedia/"
SOURCE = os.path.join(DATASET_ROOT, "val+test/rgb")
OUT_SPLITTED_VAL = os.path.join(DATASET_ROOT, "val/rgb")
OUT_SPLITTED_TEST = os.path.join(DATASET_ROOT, "test/rgb")


def moveAll(coco, val):
    coco_ids = coco.getImgIds()
    for i in tqdm(range(0, len(coco_ids))):
        img = coco.loadImgs(coco_ids)[i]
        file_name = img["kaggle_id"] + ".jpg"
        path_src = os.path.join(SOURCE, file_name)
        path_dst = (
            os.path.join(OUT_SPLITTED_VAL, file_name)
            if val
            else os.path.join(OUT_SPLITTED_TEST, file_name)
        )
        try:
            shutil.move(path_src, path_dst)
        except FileNotFoundError:
            print("Not found: " + path_src)
            continue


def main():
    val_json = os.path.join(DATASET_ROOT, "instances_attributes_val2020.json")
    test_json = os.path.join(DATASET_ROOT, "info_test2020.json")
    val_coco = COCO(val_json)
    test_coco = COCO(test_json)

    moveAll(val_coco, val=True)
    moveAll(test_coco, val=False)


if __name__ == "__main__":
    main()
