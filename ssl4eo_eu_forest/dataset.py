import os
import torch
import rasterio
import requests
import shapely
import geopandas as gpd
import pandas as pd
import logging
from torchgeo.datasets import GeoDataset
from datasets import load_dataset
from huggingface_hub import hf_hub_url
from typing import Optional, Callable, List, Dict, Any, Union

# Logger setup
logger = logging.getLogger("SSL4EOEUForestTG")
logger.setLevel(logging.DEBUG)

if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class SSL4EOEUForestTG(GeoDataset):
    """TorchGeo dataset for SSL4EO-EU Forest segmentation with seasonal imagery."""

    def __init__(
        self,
        root: str,
        repo_id: str = "dm4eo/ssl4eo_eu_forest",
        revision: str = "v1.0",
        transforms: Optional[Callable] = None
    ):
        super().__init__()
        self.root = root
        self.transforms = transforms

        logger.info(f"Loading SSL4EO-EU Forest dataset from Hugging Face: {repo_id}@{revision}")

        raw_df = pd.DataFrame([
            row for row in load_dataset(
                repo_id,
                trust_remote_code=True,
                streaming=True,
                revision=revision
            )["train"]
        ])

        raw_df["geometry"] = raw_df.bbox_epsg4326.apply(
            lambda b: shapely.geometry.box(b[0], b[1], b[2], b[3])
        )

        df = gpd.GeoDataFrame(
            raw_df.drop(columns=["bbox_epsg4326"]),
            geometry="geometry",
            crs="EPSG:4326"
        )

        df["mask_url"] = df.mask_path.apply(
            lambda mp: hf_hub_url(repo_id=repo_id, filename=mp, repo_type="dataset")
        )
        df["image_urls"] = df.images.apply(
            lambda ims: [
                hf_hub_url(repo_id=repo_id, filename=path, repo_type="dataset")
                for path in ims["path"]
            ]
        )

        self.df = df
        self.index = list(df.index)

        logger.info(f"Dataset initialized with {len(self.df)} samples")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        group_id = row["group_id"]
        mask_url = row["mask_url"]
        image_urls = row["image_urls"]
        image_meta = row["images"]

        group_dir = os.path.join(self.root, str(group_id))
        os.makedirs(group_dir, exist_ok=True)

        # Load mask
        mask_path = os.path.join(group_dir, "mask.tif")
        if not os.path.exists(mask_path):
            logger.debug(f"Downloading mask for group {group_id} from {mask_url}")
            response = requests.get(mask_url)
            response.raise_for_status()
            with open(mask_path, "wb") as f:
                f.write(response.content)
        else:
            logger.debug(f"Using cached mask for group {group_id}")

        with rasterio.open(mask_path) as src:
            mask_array = src.read()
            mask = torch.from_numpy(mask_array).byte()
            logger.debug(f"Loaded mask shape: {mask.shape}, dtype: byte")

        # Load seasonal images
        season_tensors = []
        metadata = []

        for i, url in enumerate(image_urls):
            season = image_meta["season"][i]
            timestamp_start = image_meta["timestamp_start"][i]
            timestamp_end = image_meta["timestamp_end"][i]
            tile_id = image_meta["tile_id"][i]

            image_path = os.path.join(group_dir, f"{season}.tif")
            if not os.path.exists(image_path):
                logger.debug(f"Downloading {season} image for group {group_id} from {url}")
                response = requests.get(url)
                response.raise_for_status()
                with open(image_path, "wb") as f:
                    f.write(response.content)
            else:
                logger.debug(f"Using cached {season} image for group {group_id}")

            with rasterio.open(image_path) as src:
                image_array = src.read()
                image_tensor = torch.from_numpy(image_array).to(torch.uint16)
                season_tensors.append(image_tensor)

                metadata.append({
                    "season": season,
                    "timestamp_start": timestamp_start,
                    "timestamp_end": timestamp_end,
                    "tile_id": tile_id,
                    "shape": image_tensor.shape,
                    "crs": src.crs.to_string() if src.crs else None,
                    "transform": src.transform
                })

                logger.debug(f"Loaded {season} image shape: {image_tensor.shape}, dtype: uint16")

        image = torch.stack(season_tensors, dim=0)
        logger.debug(f"Final image shape (seasonal stack): {image.shape}")

        sample = {
            "image": image,
            "mask": mask,
            "group_id": group_id,
            "metadata": metadata
        }

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        return len(self.df)

    def rgb_from_samples(self, sample_or_index: Union[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Extract RGB images and mask from a sample for visualization."""
        def normalize_band(band: torch.Tensor, lower=2, upper=98) -> torch.Tensor:
            band = band.float()
            flat = band.flatten()
            low = torch.quantile(flat, lower / 100.0)
            high = torch.quantile(flat, upper / 100.0)
            band_clipped = torch.clamp(band, min=low.item(), max=high.item())
            return (band_clipped - low) / (high - low + 1e-5)

        sample = self[sample_or_index] if isinstance(sample_or_index, int) else sample_or_index
        rgb_dict = {}

        seasonal_stack = sample["image"]  # [S, C, H, W]
        seasons = [meta["season"] for meta in sample["metadata"]]

        for i, season in enumerate(seasons):
            # Use Sentinel-2 bands 2, 3, 4 → torch indices [1, 2, 3]
            rgb_tensor = seasonal_stack[i][1:4]  # [3, H, W]
            normalized = torch.stack([normalize_band(rgb_tensor[j]) for j in range(3)], dim=0)
            rgb_np = normalized.permute(1, 2, 0).cpu().numpy()
            rgb_dict[season] = rgb_np

        mask_tensor = sample["mask"][0]
        mask_np = mask_tensor.cpu().numpy()
        rgb_dict["mask"] = mask_np

        return rgb_dict

    def show_bbox_folium(self, sample_or_index: Union[int, Dict[str, Any]]) -> "folium.Map":
        """Display the bounding box of a sample on an interactive folium map."""
        import folium
        from shapely.geometry import mapping

        if isinstance(sample_or_index, int):
            idx = sample_or_index
        else:
            idx = self.index[self.df["group_id"] == sample_or_index["group_id"]][0]

        geom = self.df.iloc[idx].geometry
        bounds = geom.bounds
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

        m = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")
        geo_json = mapping(geom)

        folium.GeoJson(geo_json, name="Bounding Box", style_function=lambda x: {
            "color": "red", "weight": 2, "fillOpacity": 0
        }).add_to(m)

        folium.LayerControl().add_to(m)
        return m
