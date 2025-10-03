# Code to support loading the SSL4EO-EU-Forest dataset

## Dataset

 Details on HuggingFace: https://huggingface.co/datasets/dm4eo/ssl4eo_eu_forest .

## Quickstart

Install the `ssl4eu_eu_forest` Python module:
```Bash
pip install git+https://github.com/cmalbrec/ssl4eo_eu_forest.git
```
and stream-load a sample through the custom TorchGeo dataset:
```Python
from ssl4eo_eu_forest import SSL4EOEUForestTG
dataset = SSL4EOEUForestTG(root="./cache")
```

Additional details provides [README.ipynb](README.ipynb).

## Funding

This work was carried under the *EvoLand* project, cf. https://www.evo-land.eu . This project has received funding from the European Union's Horizon Europe research and innovation programme under grant agreement No. `101082130`.

## Citation

```bibtex
@misc{ssl4eo_eu_forest,
  author       = {Braham Ait Ali, Nassim and Albrecht, Conrad M},
  title        = {SSL4EO-EU Forest Dataset},
  year         = {2025},
  howpublished = {https://github.com/cmalbrec/ssl4eo_eu_forest}
}
```
