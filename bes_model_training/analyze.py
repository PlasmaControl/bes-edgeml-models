from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow as tf
from tensorflow import keras
from train import Exp_Learning_Rate_Schedule

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


# train_dir = 'multitrain-01_20210421_110057'
# train_dir = 'multitrain-01_20210421_142545'
# train_dir = 'multitrain-01_20210421_142605'
# train_dir = 'multitrain-01_20210421_213026'
# train_dir = 'multitrain-01_20210422_110440'
train_dir = 'multitrain-01_20210422_110420'

model_dir = Path('models') / train_dir

model = keras.models.load_model(model_dir/'saved_model.tf',
                                custom_objects={
                                    'Exp_Learning_Rate_Schedule': Exp_Learning_Rate_Schedule,
                                    },
                                compile=True,
                                )

test_data = model_dir / 'test_data.pickle'
with test_data.open('rb') as f:
    data = pickle.load(f)

signals = data['signals']
labels = tf.cast(data['labels'], tf.float16)


n_super_windows = labels.shape[0]
super_window_size = labels.shape[1]
n_evals = super_window_size - 7
prediction = np.empty((n_super_windows, n_evals), dtype=np.float16)

print('Calculating model predictions with test dataset')
for i in range(n_evals):
    if i%25 == 0:
        print(f'  super_window index: {i}')
    data = np.array(signals[:, i:i+8, :, :]).reshape([-1,8,8,8,1])
    p = model.predict(data,
                      batch_size=24,
                      )
    prediction[:, i] = np.squeeze(p)

prediction[prediction<0] = 0.0
prediction[prediction>0] = 1.0

# labels = labels[:, 7:]

labels_flat = tf.reshape(labels[:, 7:], [-1])
prediction_flat = prediction.reshape([-1])

print('Calculating predition accuracy')
accuracy = np.array(keras.metrics.binary_accuracy(
    labels_flat,
    prediction_flat,
    ))

print('Calculating confusion matrix')
confusion_matrix = np.array(tf.math.confusion_matrix(
    labels_flat,
    prediction_flat,
    num_classes=2,
    ))

print(f'Overall accuracy: {accuracy:.3f}')
print('Confusion matrix:')
print(confusion_matrix)

ax = sn.heatmap(confusion_matrix, annot=True, fmt="d", linewidths=2, cmap='copper_r')
ax.set_title(f'Accuracy: {accuracy*1e2:.1f} %')
ax.set_xlabel('Prediction', size='large')
ax.set_ylabel('Ground truth', size='large')
ax_labels = ['No ELM', 'Active ELM']
ax.set_xticklabels(ax_labels)
ax.set_yticklabels(ax_labels)


i_active_elm = np.nonzero(np.any(labels==1, axis=1))[0]
i_inactive_elm = np.nonzero(np.all(labels==0, axis=1))[0]

np.random.shuffle(i_active_elm)
np.random.shuffle(i_inactive_elm)

fig = plt.figure(figsize=[13.5,7.25])
for i in range(12):
    plt.subplot(3,4,i+1)
    if i<6:
        i_elm = i_active_elm[i]
    else:
        i_elm = i_inactive_elm[i]
    bes_signal = np.array(signals[i_elm, :, 2, 6]).reshape(-1)
    bes_label = np.array(labels[i_elm, :]).reshape(-1)
    bes_time = np.arange(super_window_size)
    plt.plot(bes_time, bes_signal, label='BES ch. 22')
    plt.plot(bes_time, bes_label+0.02, label='Ground truth', ls='-.')
    plt.plot(bes_time[7:], prediction[i_elm,:], label='Prediction', ls='-.')
    plt.xlabel('Time (micro-s)')
    plt.ylabel('Signal | label')
    plt.ylim([None,1.1])
    plt.legend(fontsize='small')
plt.tight_layout()