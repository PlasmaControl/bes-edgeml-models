# Models
This directory contains all the neural network models. Each module is a standalone 
script containing a different model architecture. A small description of the 
modules is as follows:
- Different versions of the `feature_model.py` have the same underlying 
architecture. They vary in terms of the number of CNN filters and nodes in the 
fully connected layers. 
- Different CNN models have similar above-mentioned differences from each other 
except [`cnn_2d_model.py`](cnn_2d_model.py) where a projection along the time axis (axis=2) of the 
input tensor of shape (`batch_size`, `n_channels`, `signal_window_size`, 8, 8) is 
taken with 64 trainable kernels of size `signal_window_size`. This projection will 
compress the 3-d signal of size (`signal_window_size`, 8, 8) to a 2-d image of size 
`8x8`. The standard convolution operation can then be applied on the signals.
- [`multi_features_model.py`](multi_features_model.py) - One of the best performing 
models, incorporating features from multiple data representations like raw BES
signal, discrete (continuous) wavelet transform and fast Fourier transform. For 
longer lookaheads, it showed much better performance than any of the other models 
that were tried.
- [`rnn_model.py`](rnn_model.py) - Basic LSTM neural network heavily inspired from
sentiment analysis. __Not actively maintained__.
- [`mts_cnn_model.py`](mts_cnn_model.py) - Multi variable time series CNN, inspired from
[this article](https://link.springer.com/article/10.1007/s10845-020-01591-0) by Hsu et al.
__Not actively maintained__.
- [`feature_gradients_model.py`](feature_gradients_model.py) - A deep convolutional 
neural network using the flattened and concatenated spatial and temporal gradients
as input features to the classifier. __Not actively maintained__.
- [`multi_features_ds_model.py`](multi_features_ds_model.py) - All the code is migrated to `multi_features_model.py`. 
__Not actively maintained__.

## Important notes about [`multi_features_model.py`](multi_features_model.py)
As mentioned before, multi features model takes in features from different data 
representations. It was observed that in addition to the raw BES signals, and FFT
features, the continuous wavelet transform (CWT) really helps the model to perform great
for lookaheads as big as 1 ms, see figure below:

|  F1-score comparison between `feature` and `multifeature` model   |     F1-score for different `signal_window_size`      |
|:-----------------------------------------------------------------:|:----------------------------------------------------:|
| ![image1](../assets/f1-comparison-feature-multifeature-model.png) | ![image2](../assets/f1-comparison-different-sws.png) |

The figure above on the left shows multi-feature model has similar performance for
a lookahead of 1 ms as the feature model for a lookahead of 0 whereas the figure
on right shows multi-feature model has better performance than the feature model for
similar or larger `signal_window_size` (`sws`). Although this looks quite encouraging, 
there is a huge problem with using CWT in multi-features model - __it lets the model
peek into the future__ or in other words, there is __data-leakage__ with the way 
continuous wavelet transforms are calculated. This data leakage will be more prominent
with large scales for the CWT, see figure below _(thanks to Jeffery Zimmerman for the plots 
and pointing out this issue)_.

![image3](../assets/cwt_issues.png)

The model with data-leakage issue will show a great performance on the training 
set (overfitting), but it  will fail to generalize well on the previously unseen test data.
To remedy this issue, we created a sort of custom continuous wavelet transforms where
all the scales are stacked together to the right and padded upto `signal_window_size` 
to avoid the sliding convolution between the signal and the wavelet filter. We assigned 
the result of the dot product between wavelet filter for a given scale and the signal 
to the leading time index for the signal. For instance, if the input signal has shape
(`signal_window_size`, 8, 8) and the wavelet filter bank has shape (`n_scales`, `signal_window_size`),
the output after dot product will have shape (`n_scales`, 8, 8). The right stacked filter 
bank will look like the following figure:

![image4](../assets/stacked_wavelet_filters.png)

This approach, although seems to rectify the data leakage issue does not help in 
improving the model's performance.

Finally, we ended up using the discrete wavelet transforms (DWT) which showed better 
performance than our custom wavelet implementation. We used PyTorch implementation of the DWT
from package [pytorch_wavelets](https://github.com/fbcotter/pytorch_wavelets).

