# Code to support loading the SSL4EO-EU-Forest dataset

## Dataset

 Detail on HuggingFace: https://huggingface.co/datasets/dm4eo/ssl4eo_eu_forest .

 # Quickstart

Install the `ssl4eu_eu_forest` Python module:
```Bash
pip install git+https://github.com/cmalbrec/ssl4eo_eu_forest.git
```
and stream load a sample through the custom TorchGeo dataset:
```Python
from ssl4eo_eu_forest import SSL4EOEUForestTG
dataset = SSL4EOEUForestTG(root="./cache")
```

Additional details provides [README.ipynb](README.ipynb).
