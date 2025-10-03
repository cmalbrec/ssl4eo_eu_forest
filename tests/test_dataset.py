from ssl4eo_eu_forest.dataset import SSL4EOEUForestTG
import pytest

def test_dataset_loads():
    try:
        ds = SSL4EOEUForestTG(root="./cache")
        sample = ds[0]
        assert "image" in sample and "mask" in sample
    except RuntimeError as e:
        if "Dataset scripts are no longer supported" in str(e):
            pytest.skip("Dataset script loading is deprecated on Hugging Face Hub.")
        else:
            raise

