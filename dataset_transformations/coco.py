# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""COCO"""
import json
import os
from pathlib import Path

import datasets

_SPLIT_MAP = {"train": "train2017", "validation": "val2017"}

_CATEGORY_NAMES = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", "basketball-court", "ground-track-field",
    "harbor", "bridge", "small-vehicle", "large-vehicle", "roundabout", "swimming-pool", "helicopter",
    "soccer-ball-field", "container-crane"
]

_FEATURES = datasets.Features(
    {
        "image_id": datasets.Value("int64"),
        "image": datasets.Image(),
        "width": datasets.Value("int64"),
        "height": datasets.Value("int64"),
        "objects": datasets.Sequence({
            "bbox_id": datasets.Value("int64"),
            "category": datasets.ClassLabel(names=_CATEGORY_NAMES),
            "bbox": datasets.Sequence(datasets.Value("int64"), 4),
            "area": datasets.Value("int64"),
        })
    }
)


class COCO(datasets.GeneratorBasedBuilder):
    """COCO"""

    VERSION = datasets.Version("3.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="dota", version=VERSION, description=""
        ),
    ]

    DEFAULT_CONFIG_NAME = "dota"

    def _info(self):
        return datasets.DatasetInfo(
            features=_FEATURES,
        )

    def _split_generators(self, dl_manager):
        annotation_file = {
            "train": os.path.join("/home/hicham/Documents/datasets/dota/annotations", "instances_train2017.json"),
            "validation": os.path.join("/home/hicham/Documents/datasets/dota/annotations", "instances_val2017.json"),
            "test": os.path.join("/home/hicham/Documents/datasets/dota/annotations", "instances_test2017.json")}
        image_folders = {"train": Path("/home/hicham/Documents/datasets/dota/train2017"),
                         "validation": Path("/home/hicham/Documents/datasets/dota/val2017"),
                         "test": Path("/home/hicham/Documents/datasets/dota/test2017")}

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "annotation_file": annotation_file["train"],
                    "image_folders": image_folders,
                    "split_key": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "annotation_file": annotation_file["validation"],
                    "image_folders": image_folders,
                    "split_key": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "annotation_file": annotation_file["test"],
                    "image_folders": image_folders,
                    "split_key": "test",
                },
            ),
        ]

    def _generate_examples(self, annotation_file, image_folders, split_key):
        with open(annotation_file, "r", encoding="utf-8") as f:
            annotations = json.load(f)

            for image_metadata in annotations["images"]:
                image_path = image_folders[split_key] / image_metadata["file_name"]

                record = {
                    "image_id": image_metadata["id"],
                    "image": str(image_path.absolute()),
                    "width": image_metadata["width"],
                    "height": image_metadata["height"],
                    "objects": [{
                        "bbox_id": ann["id"],
                        "category": ann["category_id"],
                        "bbox": ann["bbox"],
                        "area": ann["area"],
                    } for ann in annotations["annotations"] if ann["image_id"] == image_metadata["id"]]
                }
                yield record["image_id"], record
