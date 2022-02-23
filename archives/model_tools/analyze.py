import pickle
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import tensorflow as tf

try:
    from . import utilities
except:
    import utilities


# TF environment
print('TF version:', tf.__version__)
for gpu_device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu_device, True)
print('Available devices:')
for device in tf.config.list_physical_devices():
    print(f'  {device.device_type}, {device.name}')

# device visibility
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[-1], 'GPU')
print('Visible devices:')
for device in tf.config.get_visible_devices():
    print(f'  {device.device_type}, {device.name}')


train_dir = 'trained_model_20210514_114832'


model_dir = utilities.model_dir / train_dir

model = tf.keras.models.load_model(model_dir/'saved_model.tf',
                                   custom_objects={
                                       'Exp_Learning_Rate_Schedule': utilities.Exp_Learning_Rate_Schedule,
                                       },
                                   compile=True,
                                   )


# read test data saved during training
test_data_file = model_dir / 'test_data.pickle'
with test_data_file.open('rb') as f:
    data = pickle.load(f)

signals = np.array(data['signals'])
labels = np.array(data['labels'])
sample_indices = np.array(data['sample_indices'])
window_start_indices = np.array(data['window_start_indices'])
signal_window_size = data['signal_window_size']
label_look_ahead = data['label_look_ahead']


def plot():
    n_elms = window_start_indices.size
    rng = np.random.default_rng()
    i_elms = rng.choice(n_elms, 12, replace=False)

    plt.close('all')

    plt.figure(figsize=[13.5,7.25])
    for i, i_elm in enumerate(i_elms):
        i_start = window_start_indices[i_elm]
        if i_elm < n_elms-1:
            i_stop = window_start_indices[i_elm+1]-1
        else:
            i_stop = labels.size
        print(f'ELM {i+1} of 12 with {i_stop-i_start+1} time points')
        elm_signals = signals[i_start:i_stop, :, :]
        elm_labels = labels[i_start:i_stop]
        predictions = np.zeros(elm_labels.size-signal_window_size-label_look_ahead+1)
        for j in range(predictions.size):
            if j%500==0:
                print(f'  Time {j}')
            input_signals = elm_signals[j:j+signal_window_size, :, :].reshape([1,signal_window_size,8,8,1])
            predictions[j] = model(input_signals, training=False)
        predictions = sp.special.expit(predictions)  # convert logits to probability
        plt.subplot(3,4,i+1)
        elm_time = np.arange(elm_labels.size)
        plt.plot(elm_time, elm_signals[:, 2, 6], label='BES ch. 22')
        plt.plot(elm_time, elm_labels+0.02, label='Ground truth', ls='-.')
        plt.plot(elm_time[(signal_window_size+label_look_ahead-1):], predictions, label='Prediction', ls='-.')
        plt.xlabel('Time (micro-s)')
        plt.ylabel('Signal | label')
        plt.ylim([None,1.1])
        plt.legend(fontsize='small')
    plt.tight_layout()


if __name__=='__main__':
    plot()