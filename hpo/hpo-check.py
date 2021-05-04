import numpy as np
import matplotlib.pyplot as plt
import optuna


db_file = '/home/dsmith/scratch/optuna/hpo-02.db'
db_url = f'sqlite:///{db_file}'
study_name = 'study'

study = optuna.load_study(storage=db_url, study_name=study_name)


print('Trials')
for trial in study.trials:
    if not trial.intermediate_values:
        print(f'  Num {trial.number:03d} in first epoch')
        continue
    if trial.value:
        value = trial.value
    else:
        value = list(trial.intermediate_values.values())[-1]
    report = f'  Num {trial.number:03d} value {value:.3f} last step {trial.last_step} state {trial.state.name}'
    if trial.state.name != 'RUNNING':
        report += f' duration {trial.duration.seconds/3600:.1f} hr'
    print(report)


importances = optuna.importance.get_param_importances(study)
print('Importances:')
for key, value in importances.items():
    print(f'  {key} importance: {value:.2f}')


best_trial = study.best_trial
print('Best trial')
print(f'  Number {best_trial.number}')
print(f'  Value: {best_trial.value:.4f}')
print(f'  Duration: {best_trial.duration.seconds/3600:.1f} hr')
print(f'  Steps: {len(best_trial.intermediate_values)}')
for key, value in best_trial.params.items():
    print(f'    {key}: {value}')


valid_trials = [trial for trial in study.trials
                if trial.state.name != 'RUNNING'
                and trial.value is not None]

values = [trial.value for trial in valid_trials]
value_percentile = np.percentile(values, 10)
best_trials = [trial for trial in valid_trials if trial.value <= value_percentile]

nparams = len(best_trial.params.keys())
ncol = nparams//3 + 1
value_data = [trial.value for trial in best_trials]


plt.close('all')

# plot parameter histograms
fig, axes = plt.subplots(ncols=ncol, nrows=3, sharey=True, figsize=(13,6.5))
for i, key in enumerate(best_trial.params.keys()):
    if key in ['l2_factor', 'initial_learning_rate',
               'relu_negative_slope', 'minimum_learning_rate_factor']:
        param_data = np.log10([trial.params[key] for trial in best_trials])
        label = f'log( {key} )'
    else:
        param_data = [trial.params[key] for trial in best_trials]
        label = key
    plt.sca(axes.flat[i])
    # plt.plot(param_data, value_data, 'x')
    plt.hist(param_data)
    plt.xlabel(label)
    # plt.ylabel('Binary cross-entropy')
    plt.title(label)
plt.tight_layout()

# plot training histories
plt.figure()
for trial in best_trials:
    plt.plot(trial.intermediate_values.values(), label=f'{trial.number}')
plt.xlabel('Epoch')
plt.ylabel('Binary cross-entropy')
plt.title('Training intermediate values')
plt.tight_layout