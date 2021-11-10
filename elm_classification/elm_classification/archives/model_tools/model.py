"""
TF models
"""
from typing import Tuple, Union
import tensorflow as tf


tf.keras.mixed_precision.set_global_policy("mixed_float16")


def fully_connected_layers(
    x,
    dense_layers: Union[int, Tuple[int, int]] = (40, 20),
    dropout_rate: float = 0.2,
    l2_factor: float = 5e-3,
    relu_negative_slope: float = 0.02,
) -> tf.Tensor:
    """Creates a neural network with few fully connected layers.

    Args:
    -----
        x (tf.Tensor): Input to the network.
        dense_layers (Union[int, Tuple[int, int]], optional): Number of hidden
            units in each layer. Defaults to (40, 20).
        dropout_rate (float, optional): Fraction of total hidden units that will
            be turned off for drop out. Defaults to 0.2.
        l2_factor (float, optional): Lambda for l2-regularization. Defaults to 5e-3.
        relu_negative_slope (float, optional): Slope of ReLU activation for negative
            `x`. Defaults to 0.02.

    Returns:
    --------
        tf.Tensor: Tensor containing the output logits.
    """

    # flatten
    x = tf.keras.layers.Flatten()(x)
    print(f"  Flattening tensors; output shape: {x.shape}")

    # add fully-connected layers with dropout and regularizers
    for _, layer_size in enumerate(dense_layers):
        x = tf.keras.layers.Dense(
            layer_size,
            activation=tf.keras.layers.ReLU(negative_slope=relu_negative_slope),
            kernel_regularizer=tf.keras.regularizers.l2(l2_factor),
            bias_regularizer=tf.keras.regularizers.l2(l2_factor),
        )(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        print(
            f"  Applying dense layer with size {layer_size}; output shape: {x.shape}"
        )

    # final binary classification
    print("  Output is logit, not probability")
    x = tf.keras.layers.Dense(1)(x)
    return x


def cnn_model(
    signal_window_size: int = 8,
    dense_layers: Union[int, Tuple[int, int]] = (40, 20),
    dropout_rate: float = 0.2,
    l2_factor: float = 5e-3,
    relu_negative_slope: float = 0.02,
    conv_size: int = 3,
    cnn_layers: Tuple[int, int] = (4, 8),
) -> tf.keras.Model:
    """
    CNN layers followed by fully-connected layers.

    Args:
    -----
        signal_window_size (int, optional): Number of time data points to be used to create
            input 3-d tensor. Defaults to 8.
        dense_layers (Union[int, Tuple[int, int]], optional): Number of hidden
            units in each layer. Defaults to (40, 20).
        dropout_rate (float, optional): Fraction of total hidden units that will
            be turned off for drop out. Defaults to 0.2.
        l2_factor (float, optional): Lambda for l2-regularization. Defaults to 5e-3.
        relu_negative_slope (float, optional): Slope of ReLU activation for negative
            `x`. Defaults to 0.02.
        conv_size (int, optional): Size of the convolution kernel/filter. Defaults
            to 3.
        cnn_layers (Tuple[int, int], optional): Dimensionality of the output space.
            Essentially, it gives the number of output kernels after convolution.
            Defaults to (4, 8).

    Returns:
    --------
        tf.keras.Model: Model instance.
    """

    # input layer: n_lookback time points, 8x8 BES grid, 1 "channel"
    inputs = tf.keras.Input(shape=(signal_window_size, 8, 8, 1))
    x = inputs
    print(f"Input layer shape: {x.shape}")

    # apply cnn layers
    for i_layer, filters in enumerate(cnn_layers):
        if i_layer == 0:
            filter_shape = (signal_window_size, conv_size, conv_size)
        else:
            filter_shape = (1, conv_size, conv_size)
        if filters == 0:
            continue
        # apply conv. layer and dropout
        x = tf.keras.layers.Conv3D(
            filters,
            filter_shape,
            activation=tf.keras.layers.ReLU(negative_slope=relu_negative_slope),
            kernel_regularizer=tf.keras.regularizers.l2(l2_factor),
            bias_regularizer=tf.keras.regularizers.l2(l2_factor),
        )(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        print(
            f"  Applying {filters} conv. filters with shape {filter_shape}; output shape: {x.shape}"
        )

    # fully-connected layers
    x = fully_connected_layers(
        x,
        dense_layers=dense_layers,
        dropout_rate=dropout_rate,
        l2_factor=l2_factor,
        relu_negative_slope=relu_negative_slope,
    )

    print(f"  Final output shape: {x.shape}")

    model = tf.keras.Model(inputs=inputs, outputs=x, name="BES_CNN_Model")
    model.summary()
    return model


def feature_model(
    signal_window_size: int = 8,
    dense_layers: Union[int, Tuple[int, int]] = (40, 20),
    dropout_rate: float = 0.2,
    l2_factor: float = 5e-3,
    relu_negative_slope: float = 0.02,
    maxpool_size: int = 2,
    filters: int = 10,
) -> tf.keras.Model:
    """
    8x8 + time feature blocks followed by fully-connected layers. This function
    takes in a 4-dimensional tensor of size: `(signal_window_size, 8, 8, 1)`
    performs maxpooling to downsample the spatial dimension by half, perform a
    3-d convolution with a filter size identical to the spatial dimensions of the
    input to avoid the sliding of the kernel over the input. Finally, it adds a
    couple of fully connected layers on the output of the 3-d convolution.

    Args:
    -----
        signal_window_size (int, optional): Number of time data points to be used to create
            input 3-D tensor. Defaults to 8.
        dense_layers (Union[int, Tuple[int, int]], optional): Number of hidden
            units in each layer. Defaults to (40, 20).
        dropout_rate (float, optional): Fraction of total hidden units that will
            be turned off for drop out. Defaults to 0.2.
        l2_factor (float, optional): Lambda for l2-regularization. Defaults to 5e-3.
        relu_negative_slope (float, optional): Slope of ReLU activation for negative
            `x`. Defaults to 0.02.
        maxpool_size (int, optional): Size of the kernel used for maxpooling. Use
            0 to skip maxpooling. Defaults to 2.
        filters (int, optional): Dimensionality of the output space.
            Essentially, it gives the number of output kernels after convolution.
            Defaults to 10.

    Returns:
    --------
        tf.keras.Model: Model instance.
    """

    # input layer: 8x8 BES grid, n_lookback time points, 1 "channel"
    inputs = tf.keras.Input(shape=(signal_window_size, 8, 8, 1))
    x = inputs
    print(f"Input layer shape: {x.shape}")

    # maxpool over spatial dimensions
    if maxpool_size:
        assert 8 % maxpool_size == 0  # size must evenly divide 8
        x = tf.keras.layers.MaxPool3D(
            pool_size=[1, maxpool_size, maxpool_size],
        )(x)
        print(
            f"  Applying spatial MaxPool with size {maxpool_size}; output shape: {x.shape}"
        )

    # full-size "convolution" layer, so the kernel does not slide or shift
    filter_shape = x.shape[1:4]
    x = tf.keras.layers.Conv3D(
        filters,
        filter_shape,
        strides=filter_shape,
        activation=tf.keras.layers.ReLU(negative_slope=relu_negative_slope),
        kernel_regularizer=tf.keras.regularizers.l2(l2_factor),
        bias_regularizer=tf.keras.regularizers.l2(l2_factor),
    )(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    print(
        f"  Applying {filters} filter kernels with shape {filter_shape}; output shape: {x.shape}"
    )

    # fully-connected layers
    x = fully_connected_layers(
        x,
        dense_layers=dense_layers,
        dropout_rate=dropout_rate,
        l2_factor=l2_factor,
        relu_negative_slope=relu_negative_slope,
    )

    print(f"  Final output shape: {x.shape}")

    model = tf.keras.Model(inputs=inputs, outputs=x, name="BES_Feature_Model")
    model.summary()
    return model


if __name__ == "__main__":

    # turn off GPU visibility
    tf.config.set_visible_devices([], "GPU")

    print("TF version:", tf.__version__)
    print("Visible devices:")
    for device in tf.config.get_visible_devices():
        print(f"  {device.device_type} {device.name}")

    test_model_1 = cnn_model()
    test_model_2 = feature_model()
