import json
import datasets
from datasets.utils.file_utils import xopen

class SSL4EOEUForest(datasets.GeneratorBasedBuilder):
    """
    Metadata generator for the SSL4EO-EU-Forest dataset, cf. https://huggingface.co/datasets/dm4eo/ssl4eo_eu_forest .
    """
    def _info(self):
        """
        Provides details on metadata structure, citation, and credits.
        """
        return datasets.DatasetInfo(
            description="SSL4EO-EU Forest dataset metadata",
            features=datasets.Features({
                # data sample ID
                "group_id": datasets.Value("string"),
                # relative path (without HuggingFace URL) of forest mask
                "mask_path": datasets.Value("string"),
                # got bounding box in lat-lon coords
                "bbox_epsg4326": datasets.Sequence(datasets.Value("float32")),
                # image dimensions in width and height
                "mask_width": datasets.Value("int32"),
                "mask_height": datasets.Value("int32"),
                # do the above dimensions match for all the images?
                "dimensions_match": datasets.Value("bool"),
                # 12-band Sentinel-2 L2A cloud-free images for all seasons in bounding box 
                "images": datasets.Sequence({
                    # relative path (without HuggingFace URL) of Sentinel-2 imagery
                    "path": datasets.Value("string"),
                    # start time for data recording
                    "timestamp_start": datasets.Value("string"),
                    # end time for data recording
                    "timestamp_end": datasets.Value("string"),
                    # Sentinel-2 tile ID
                    "tile_id": datasets.Value("string"),
                    # season in northern hemisphere
                    "season": datasets.Value("string"),
                    # image dimensions
                    "width": datasets.Value("int32"),
                    "height": datasets.Value("int32")
                })
            }),
            # which keys refer to (input, output) data for supervised
            supervised_keys=("images", "mask_path"),
            # BibTeX on how to cite this work
            citation="""@misc{ssl4eo_eu_forest,
author = {Nassim Ait Ali Braham and Conrad M Albrecht},
title = {SSL4EO-EU Forest Dataset},
year = {2025},
howpublished = {https://huggingface.co/datasets/dm4eo/ssl4eo_eu_forest},
note = {This work was carried under the EvoLand project, cf. https://www.evo-land.eu . This project has received funding from the European Union's Horizon Europe research and innovation programme under grant agreement No. 101082130.}
}""",
            # project homepage
            homepage="https://www.evo-land.eu",
            # data license
            license="CC-BY-4.0",
        )

    def _split_generators(self, dl_manager):
        """
        Define dataset splits - single "training" split for now.
        """
        url = f"{dl_manager._base_path}/meta.jsonl"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"url": url},
            )
        ]

    def _generate_examples(self, url):
        """
        Streaming-compliant serving of metadata for SSL4EO data samples.
        """
        with xopen(url, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                yield idx, json.loads(line)


from datasets import Features, Value, Sequence

def features_to_croissant(features: Features):
    """
    Convert a Hugging Face dataset Features object into a Croissant-compatible description.
    """
    def convert_feature(name: str, feature):
        if isinstance(feature, Value):
            return {
                "name": name,
                "dataType": feature.dtype,
                "description": f"{name} field",
                "isArray": False,
            }
        elif isinstance(feature, Sequence):
            inner = feature.feature
            # Handle nested sequence of Features
            if isinstance(inner, Features):
                return {
                    "name": name,
                    "isArray": True,
                    "description": f"{name} sequence",
                    "features": [convert_feature(k, v) for k, v in inner.items()]
                }
            # Handle flat sequence of Values
            elif isinstance(inner, Value):
                return {
                    "name": name,
                    "isArray": True,
                    "description": f"{name} sequence",
                    "dataType": inner.dtype
                }
        elif isinstance(feature, Features):
            # Top-level nested structure (not a sequence)
            return {
                "name": name,
                "description": f"{name} nested structure",
                "features": [convert_feature(k, v) for k, v in feature.items()],
                "isArray": False
            }
        else:
            return {
                "name": name,
                "dataType": "unknown",
                "description": f"{name} field",
                "isArray": False,
            }

    return [convert_feature(name, feature) for name, feature in features.items()]

