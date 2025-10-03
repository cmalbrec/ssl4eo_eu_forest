import pytest
import ssl4eo_eu_forest.ssl4eo_eu_forest as ssl4eo_eu_forest
import datasets

def test_info_structure():
    info = ssl4eo_eu_forest.SSL4EOEUForest()._info()
    assert isinstance(info, datasets.DatasetInfo)
    assert "group_id" in info.features
    assert "images" in info.features
    assert info.description.startswith("SSL4EO-EU Forest")
    assert info.citation.startswith("@misc{ssl4eo_eu_forest")

def test_features_to_croissant_basic():
    info = ssl4eo_eu_forest.SSL4EOEUForest()._info()
    croissant = ssl4eo_eu_forest.features_to_croissant(info.features)
    assert isinstance(croissant, list)
    assert any(f["name"] == "group_id" for f in croissant)
    assert any(f["name"] == "images" and f["isArray"] for f in croissant)

def test_convert_nested_sequence():
    features = datasets.Features({
        "images": datasets.Sequence({
            "path": datasets.Value("string"),
            "season": datasets.Value("string")
        })
    })
    croissant = ssl4eo_eu_forest.features_to_croissant(features)
    assert croissant[0]["name"] == "images"
    assert croissant[0]["isArray"] is True
    assert "features" in croissant[0]
    assert any(f["name"] == "path" for f in croissant[0]["features"])

def test_generate_examples_streaming(tmp_path):
    # Create a mock JSONL file
    jsonl = tmp_path / "meta.jsonl"
    sample = {
        "group_id": "sample_001",
        "mask_path": "mask/sample_001.tif",
        "bbox_epsg4326": [6.0, 50.0, 6.1, 50.1],
        "mask_width": 264,
        "mask_height": 264,
        "dimensions_match": True,
        "images": [{
            "path": "images/sample_001_spring.tif",
            "timestamp_start": "2022-03-01",
            "timestamp_end": "2022-03-31",
            "tile_id": "T32UQD",
            "season": "spring",
            "width": 264,
            "height": 264
        }]
    }
    with open(jsonl, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample) + "\n")

    gen = ssl4eo_eu_forest.SSL4EOEUForest()
    examples = list(gen._generate_examples(str(jsonl)))
    assert len(examples) == 1
    idx, data = examples[0]
    assert data["group_id"] == "sample_001"
    assert data["images"][0]["season"] == "spring"
