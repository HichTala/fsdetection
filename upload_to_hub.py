from dataset_transformations.coco import COCO
from datasets import load_dataset

builder = COCO()
builder.download_and_prepare()

dataset = builder.as_dataset()
dataset.push_to_hub("HichTala/dota")
