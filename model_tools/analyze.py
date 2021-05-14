import pickle
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow as tf
import sklearn.metrics

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


# train_dir = 'hpo-02_trial_119_20210504_113150/'
# train_dir = 'hpo-02_trial_013_20210502_133426/'
# train_dir = 'hpo-02_trial_083_20210503_151706/'
# train_dir = 'hpo-02_trial_141_20210504_224158/'
# train_dir = 'hpo-02_trial_089_20210503_180059/'
# train_dir = 'hpo_features_01_trial_016_20210506_030617'
train_dir = 'hpo_features_01_trial_125_20210507_175056'


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
window_start_indices = np.array(data['window_start_indices'])
signal_window_size = data['signal_window_size']


labels_trimmed = labels[:, (signal_window_size-1):]

# data shape parameters
n_super_windows = labels.shape[0]
super_window_size = labels.shape[1]
n_valid_t0 = super_window_size - (signal_window_size-1)


# evaluate model on test data
print('Calculating model predictions with test dataset')
prediction = np.empty_like(labels_trimmed)
for i in range(n_valid_t0):
    if i%50 == 0:
        print(f'  super_window index: {i}')
    data = signals[:, i:i+signal_window_size, :, :].reshape([-1,signal_window_size,8,8,1])
    pred_tmp = model.predict(data, batch_size=16)
    prediction[:, i] = np.squeeze(pred_tmp)

prediction = sp.special.expit(prediction)

print('Test dataset results')

# calcualte cross entropy
cross_entropy = tf.keras.losses.binary_crossentropy(
    labels_trimmed,
    prediction,
    from_logits=False,
)
cross_entropy = np.mean(cross_entropy)
print(f'  Cross entropy: {cross_entropy:.4f}')

# recast as integer classes
labels_classes = np.zeros(labels_trimmed.shape, dtype=np.int)
labels_classes[labels_trimmed >= 0.5] = 1
prediction_classes = np.zeros(prediction.shape, dtype=np.int)
prediction_classes[prediction >= 0.5] = 1

# calculate accuracy
accuracy = tf.keras.metrics.binary_accuracy(
    labels_classes,
    prediction_classes,
)
accuracy = np.mean(accuracy)
print(f'  Accuracy: {accuracy*1e2:.2f} %')

# calculate confusion matrix
confusion_matrix = tf.math.confusion_matrix(
    labels_classes.flat,
    prediction_classes.flat,
    num_classes=2,
)
confusion_matrix = np.array(confusion_matrix)
print('  Confusion matrix:')
print(confusion_matrix)


# ROC and PR curves
fp, tp, _ = sklearn.metrics.roc_curve(labels_classes.flat,
                                      prediction.flat)
roc_auc = sklearn.metrics.roc_auc_score(labels_classes.flat,
                                        prediction.flat)
precision, recall, _ = sklearn.metrics.precision_recall_curve(labels_classes.flat,
                                                              prediction.flat)
pr_auc = sklearn.metrics.auc(recall, precision)


def do_plot():

    plt.close('all')

    ax = sn.heatmap(confusion_matrix, annot=True, fmt="d", linewidths=2, cmap='copper_r')
    ax.set_title(f'Accuracy: {accuracy*1e2:.1f} %')
    ax.set_xlabel('Prediction', size='large')
    ax.set_ylabel('Ground truth', size='large')
    ax_labels = ['No ELM', 'Active ELM']
    ax.set_xticklabels(ax_labels)
    ax.set_yticklabels(ax_labels)

    i_active_elm = np.nonzero(np.any(labels >= 0.5, axis=1))[0]
    i_inactive_elm = np.nonzero(np.all(labels < 0.5, axis=1))[0]
    np.random.shuffle(i_active_elm)
    np.random.shuffle(i_inactive_elm)

    plt.figure(figsize=[13.5,7.25])
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
        plt.plot(bes_time[(signal_window_size-1):], prediction[i_elm,:], label='Prediction', ls='-.')
        plt.xlabel('Time (micro-s)')
        plt.ylabel('Signal | label')
        plt.ylim([None,1.1])
        plt.legend(fontsize='small')
    plt.tight_layout()

    plt.figure(figsize=(6.5,3))
    plt.subplot(121)
    plt.plot(fp, tp)
    plt.xlabel('TP/P (Sens.)')
    plt.ylabel('FP/N')
    plt.title('ROC - Inactive ALM')
    plt.annotate(f'AUC = {roc_auc:.3f}', (0.5, 0.42))
    plt.grid()
    plt.subplot(122)
    plt.plot(recall, precision)
    plt.xlabel('TP/P (Recall)')
    plt.ylabel('TP/pred. pos. (Prec.)')
    plt.title('Precision/recall')
    plt.annotate(f'AUC = {pr_auc:.3f}', (0.1,0.42))
    plt.grid()
    plt.tight_layout()


if __name__=='__main__':
    do_plot()