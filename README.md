# BES ML

ML models for DIII-D BES data

To use this repo, clone it and add the repo directory to `$PYTHONPATH`.  To contribute to this repo, branch off of `main`, push the feature branch to Github, and submit PRs.  Prior to submitting PRs, pull and merge any updates from `main` branch and run test scripts.

`bes_ml/` contains modules and classes to create, train, and analyze BES ML models.  `bes_ml/base/` contains the base code, and other directories under `bes_ml/` contain specific applications that import `bes_ml/base/`.  Each application directory should contain `train.py` and `analyze.py` modules.  Example usage:

```python
from bes_ml.elm_classification.train import ELM_Classification_Trainer

model = ELM_Classification_Trainer()
model.train()
```

Other exmaples are in `if __name__ ...` blocks in `train.py` modules.

`bes_data/` contains small sample datasets (~10 MB HDF5 files) to assist with code development and tools to package BES data on the GA cluster.  `test/` contains pytest tests, and additional examples can be inferred from the test scripts.