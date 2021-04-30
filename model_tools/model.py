import tensorflow as tf


tf.keras.mixed_precision.set_global_policy('mixed_float16')


def fully_connected_layers(
        x,
        dense_layers=(40, 20),
        dropout_rate=0.2,
        l2_factor=5e-3,
        relu_negative_slope=0.02,
        use_sigmoid=False,
        ):
    # add fully-connected layers with dropout
    for i_layer, layer_size in enumerate(dense_layers):
        if layer_size == 0:
            continue
        x = tf.keras.layers.Dense(
            layer_size,
            activation=tf.keras.layers.ReLU(negative_slope=relu_negative_slope),
            kernel_regularizer=tf.keras.regularizers.l2(l2_factor),
            bias_regularizer=tf.keras.regularizers.l2(l2_factor),
            )(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        print(f'  Applying dense layer with size {layer_size}; output shape: {x.shape}')
    # final binary classification
    x = tf.keras.layers.Dense(1)(x)
    # sigmoid to convert logit to probability
    if use_sigmoid:
        print('  Applying sigmoid activiation to convert logit to prob.')
        x = tf.keras.activations.sigmoid(x)
    else:
        print('  Output is logit')
    return x


def cnn_model(
        signal_window_size=8,
        conv_size=3,
        cnn_layers=(4, 8),
        dense_layers=(40, 20),
        dropout_rate=0.2,
        l2_factor=5e-3,
        relu_negative_slope=0.02,
        use_sigmoid=False,
        ):
    """
    2-layer CNN followed by FC layers
    """

    # input layer: n_lookback time points, 8x8 BES grid, 1 "channel"
    inputs = tf.keras.Input(shape=(signal_window_size, 8, 8, 1))
    x = inputs
    print(f'Input layer shape: {x.shape}')

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
        print(f'  Applying {filters} conv. filters with shape {filter_shape}; output shape: {x.shape}')

    # flatten
    x = tf.keras.layers.Flatten()(x)
    print(f'  Flattening tensors; output shape: {x.shape}')

    # fully-connected layers
    x = fully_connected_layers(
        x,
        dense_layers=dense_layers,
        dropout_rate=dropout_rate,
        l2_factor=l2_factor,
        relu_negative_slope=relu_negative_slope,
        use_sigmoid=use_sigmoid,
        )

    print(f'  Final output shape: {x.shape}')

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.summary()
    return model


def feature_model(
        signal_window_size = 8,
        pool_size = 2,
        filters=10,
        dense_layers=(40, 20),
        dropout_rate = 0.2,
        l2_factor=5e-3,
        relu_negative_slope=0.02,
        use_sigmoid=False,
        ):

    # input layer: 8x8 BES grid, n_lookback time points, 1 "channel"
    inputs = tf.keras.Input(shape=(signal_window_size, 8, 8, 1))
    x = inputs
    print(f'Input layer shape: {x.shape}')

    # maxpool over spatial dimensions
    if pool_size:
        assert(8 % pool_size == 0)
        x = tf.keras.layers.MaxPool3D(
            pool_size=[1, pool_size, pool_size],
            )(x)
        print(f'  Applying spatial MaxPool with size {pool_size}; output shape: {x.shape}')

    # full-size "convolution" layer
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
    print(f'  Applying {filters} filter kernels with shape {filter_shape}; output shape: {x.shape}')

    # flatten
    x = tf.keras.layers.Flatten()(x)
    print(f'  Flattening tensors; output shape: {x.shape}')

    # fully-connected layers
    x = fully_connected_layers(
        x,
        dense_layers=dense_layers,
        dropout_rate=dropout_rate,
        l2_factor=l2_factor,
        relu_negative_slope=relu_negative_slope,
        use_sigmoid=use_sigmoid,
        )

    print(f'  Final output shape: {x.shape}')

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.summary()
    return model


if __name__=='__main__':

    # make last GPU visible and limit GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[-1], 'GPU')

    print('TF version:', tf.__version__)

    print('Visible devices:')
    for device in tf.config.get_visible_devices():
        print(f'  {device.device_type} {device.name}')

    # test_model = cnn_model()
    test_model = feature_model()