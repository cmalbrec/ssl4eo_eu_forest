from setuptools import setup, find_packages

setup(
    name="ssl4eo_eu_forest",
    version="1.0",
    description="TorchGeo-compatible dataset for SSL4EO-EU Forest segmentation",
    author="Conrad M Albrecht",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchgeo",
        "rasterio",
        "geopandas",
        "shapely",
        "folium",
        "requests",
        "datasets",
        "huggingface_hub"
    ],
    python_requires=">=3.8",
)
