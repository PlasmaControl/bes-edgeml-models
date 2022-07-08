# Helper Scripts
Mostly contains helper modules for training and evaluation. A brief explanation 
of the scripts in the current directory:

- [`dataset.py`](dataset.py) - Module to create PyTorch dataset for training and 
inference. Other than the command line arguments and a logger object, it expects 
`train_data` or `valid_data` - a tuple return from the data preprocessing pipeline 
containing `signals`, `labels`, `valid_indices`, and `window_start_indices`.
- [`trainer.py`](trainer.py) - Boilerplate code for training and evaluation.
- [`utils.py`](utils.py) - Various utility functions for data preprocessing, 
training and validation.
- [`classical_ml`](classical_ml.py) - Training module for training classical machine 
learning algorithms like logistic regression, random forests and gradient boosting
(XGBoost) on the features obtained from different approaches like max and average 
pooling (see [`tabular_features.py`](elm_classification/archives/unused_scripts/tabular_features.py)) 
or the output feature map from the convolution layers from DNN (see 
[`feature_extractor.py`](elm_classification/feature_extractor.py)). It expects the
features in tabular form. This script is __not actively maintained__.
- [`roc_diff_models.py`](roc_diff_models.py) - Module to create ROC plots for 
comparison between different models or to assess a model's performance for different 
lookaheads. __Not actively maintained__, needs a lot of refactoring.