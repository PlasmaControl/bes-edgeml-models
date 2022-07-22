# BES ML

ML models for DIII-D BES data

To use this repo, clone it and add the repo directory to `$PYTHONPATH`.  To contribute to this repo, branch off of `main`, push the feature branch to Github, and submit PRs.

`bes_ml/` contains modules and classes to create, train, and analyze BES ML models.  `main/` contains the base code, and other directories are specific applications that use `main/`.  Example usage:

```python
from bes_ml.elm_classification import ELM_Classification_Trainer

model = ELM_Classification_Trainer()
model.train()
```

`bes_data/` contains tools to package BES data and small sample datasets.  `test/` contains pytest tests.