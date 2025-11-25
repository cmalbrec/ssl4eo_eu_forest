from setuptools import setup, find_packages

setup(
    name="ssl4eo_eu_forest",
    version="1.0",
    description="TorchGeo-compatible dataset for SSL4EO-EU Forest segmentation",
    author="Conrad M Albrecht",
    packages=find_packages(),
    install_requires=[
     "torch>=1.12",
        "torchgeo>=0.4",
        "rasterio>=1.3",
        "geopandas>=0.13",
        "shapely>=2.0",
        "folium>=0.14",
        "requests>=2.28",
        "datasets>=2.14,<=2.19.2",
        "huggingface_hub>=0.17",
        "tqdm>=4.66",
        "numpy>=1.23"
    ],
    python_requires=">=3.8",
)
