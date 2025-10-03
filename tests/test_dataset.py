from ssl4eo_eu_forest.dataset import SSL4EOEUForestTG

def test_dataset_loads():
    ds = SSL4EOEUForestTG(root="./cache")
    sample = ds[0]
    assert "image" in sample and "mask" in sample
