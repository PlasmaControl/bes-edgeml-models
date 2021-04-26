import tensorflow as tf
from tensorflow import keras


def cnn_model(
        n_lookback=8,
        n_filters_1=4,
        n_filters_2=8,
        n_dense_1=40,
        n_dense_2=20,
        spatial_conv_size=3,
        dropout_rate=0.2,
        l2_factor=5e-3,
        relu_negative_slope=0.02,
        ):

    keras.mixed_precision.set_global_policy('mixed_float16')

    # input layer: 8x8 BES grid, n_lookback time points, 1 "channel"
    inputs = keras.Input(shape=(n_lookback, 8, 8, 1))
    x = inputs

    # conv. layer #1
    filter_1_shape = (n_lookback, spatial_conv_size, spatial_conv_size)
    conv1 = keras.layers.Conv3D(
        n_filters_1,
        filter_1_shape,
        activation=keras.layers.ReLU(negative_slope=relu_negative_slope),
        kernel_regularizer=keras.regularizers.l2(l2_factor),
        bias_regularizer=keras.regularizers.l2(l2_factor),
        )
    x = conv1(x)
    print(f'Filter 1 shape {filter_1_shape} count {n_filters_1} params {conv1.count_params()}')

    # dropout for regularization
    x = keras.layers.Dropout(dropout_rate)(x)

    # conv. layer #2
    filter_2_shape = (1, spatial_conv_size, spatial_conv_size)
    conv2 = keras.layers.Conv3D(
        n_filters_2,
        filter_2_shape,
        activation=keras.layers.ReLU(negative_slope=relu_negative_slope),
        kernel_regularizer=keras.regularizers.l2(l2_factor),
        bias_regularizer=keras.regularizers.l2(l2_factor),
        )
    x = conv2(x)
    print(f'Filter 2 shape {filter_2_shape} count {n_filters_2} params {conv2.count_params()}')
    
    # dropout for regularization
    x = keras.layers.Dropout(dropout_rate)(x)

    # flatten
    x = keras.layers.Flatten()(x)

    # FC layer #1
    dense1 = keras.layers.Dense(
        n_dense_1,
        activation=keras.layers.ReLU(negative_slope=relu_negative_slope),
        kernel_regularizer=keras.regularizers.l2(l2_factor),
        bias_regularizer=keras.regularizers.l2(l2_factor),
        )
    x = dense1(x)

    # dropout for regularization
    x = keras.layers.Dropout(dropout_rate)(x)

    # FC layer #2
    dense2 = keras.layers.Dense(
        n_dense_2,
        activation=keras.layers.ReLU(negative_slope=relu_negative_slope),
        kernel_regularizer=keras.regularizers.l2(l2_factor),
        bias_regularizer=keras.regularizers.l2(l2_factor),
        )
    x = dense2(x)

    # dropout for regularization
    x = keras.layers.Dropout(dropout_rate)(x)

    # final binary classification
    outputs = keras.layers.Dense(1)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


if __name__=='__main__':
    print('TF version:', tf.__version__)

    print('Available devices:')
    for device in tf.config.list_physical_devices():
        print(f'  {device.device_type}, {device.name}')

    print('Visible devices:')
    for device in tf.config.get_visible_devices():
        print(f'  {device.device_type}, {device.name}')
        if device.device_type == 'GPU':
            tf.config.experimental.set_memory_growth(device, True)

    test_model = cnn_model()