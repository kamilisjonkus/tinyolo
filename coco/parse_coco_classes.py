import os
import json
import shutil

def parse_coco_classes(images_dir: str, annotations_dir: str, dataset_dir: str, classes_str: str):
  classes = [_cls for _cls in classes_str.split(";") if _cls != ""]
  os.makedirs(os.path.join(dataset_dir, "annotations"), exist_ok=True)
  os.makedirs(os.path.join(dataset_dir, "train"), exist_ok=True)
  os.makedirs(os.path.join(dataset_dir, "val"), exist_ok=True)

  for dataset in ["train", "val"]:
    with open(os.path.join(annotations_dir, f"instances_{dataset}2017.json"), "r") as f: coco_data = json.load(f)
    category_ids = [cat["id"] for cat in coco_data["categories"] if cat["name"] in classes]
    if not category_ids: raise ValueError(f"No valid classes found in {dataset} dataset")
    image_ids = set([ann["image_id"] for ann in coco_data["annotations"] if ann["category_id"] in category_ids])
    filtered_images = [img for img in coco_data["images"] if img["id"] in image_ids]
    src_dir = os.path.join(images_dir, f"{dataset}2017")
    dst_dir = os.path.join(dataset_dir, f"{dataset}")
    for img in filtered_images: shutil.copy(os.path.join(src_dir, img["file_name"]), os.path.join(dst_dir, img["file_name"]))
    filtered_data = {
      "info": coco_data["info"],
      "licenses": coco_data.get("licenses", []),
      "images": filtered_images,
      "annotations": [ann for ann in coco_data["annotations"] if ann["category_id"] in category_ids],
      "categories": [cat for cat in coco_data["categories"] if cat["id"] in category_ids]
    }
    with open(os.path.join(dataset_dir, "annotations", f"{dataset}.json"), "w") as f: json.dump(filtered_data, f)

if __name__ == "__main__":
  parse_coco_classes(images_dir="/home/username/datasets/", 
                     annotations_dir="/home/username/datasets/annotations_trainval2017",
                     dataset_dir="/home/username/datasets/cat_dog", classes_str="cat;dog")