from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import optuna


def plot_study(
        db_file,
        save=False,
        block=True,
):
    db_file = Path(db_file)
    print(f'Opening RDB: {db_file.as_posix()}')

    db_url = f'sqlite:///{db_file.as_posix()}'

    study = optuna.load_study('study', db_url)
    trials = study.get_trials(
        states=(optuna.trial.TrialState.COMPLETE,),
    )
    print(f'Completed trials: {len(trials)}')

    params = tuple(trials[0].params.keys())
    n_params = len(params)

    distribution_limits = {}
    for param in params:
        low = np.amin([trial.distributions[param].low for trial in trials])
        high = np.amax([trial.distributions[param].high for trial in trials])
        distribution_limits[param] = np.array([low, high], dtype=int)

    top_quantile = np.quantile([trial.value for trial in trials], 0.8)
    top_trials = [trial for trial in trials if trial.value >= top_quantile]

    ncols = 4
    nrows = n_params // ncols if n_params % ncols == 0 else (n_params // ncols) + 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3.5, nrows * 2.625))

    for i_param, param in enumerate(params):

        p_lim = distribution_limits[param]
        p_axis = np.arange(p_lim[0], p_lim[1] + 1)

        mean = np.empty(0)
        lb = np.empty(0)
        ub = np.empty(0)
        plt.sca(axes.flat[i_param])
        max_value = np.max(np.array([trial.value for trial in trials]))
        for p_value in p_axis:
            values = np.array([trial.value for trial in trials if trial.params[param] == p_value])
            for value in values:
                plt.plot(p_value, value, c='0.6', ms=2, marker='o', mfc=None)
            values = np.array([trial.value for trial in top_trials if trial.params[param] == p_value])
            for value in values:
                plt.plot(p_value, value, 'ok', ms=4)
            if values.size == 0:
                mean = np.append(mean, np.NaN)
                lb = np.append(lb, np.NaN)
                ub = np.append(ub, np.NaN)
                continue
            mean = np.append(mean, np.mean(values))
            if values.size >= 3:
                std = np.std(values)
                lb = np.append(lb, std)
                ub = np.append(ub, std)
            else:
                lb = np.append(lb, mean[-1] - values.min())
                ub = np.append(ub, values.max() - mean[-1])
        plt.errorbar(p_axis, mean, yerr=(lb, ub),
                     marker='s', capsize=6, capthick=1.5,
                     ms=6, lw=1.5, elinewidth=1.5,
                     zorder=1)
        plt.xlabel(param)
        plt.ylabel('Objective value')
        plt.ylim([max_value * 0.5, max_value])
    plt.tight_layout()
    if save:
        filepath = db_file.parent.parent / (db_file.stem + '.pdf')
        print(f'Saving file: {filepath.as_posix()}')
        plt.savefig(filepath, transparent=True)
    plt.show(block=block)

if __name__=='__main__':
    # db_name = 'optuna_la800_sws512'
    # db_file = Path.home() / 'scratch/edgeml/work' / db_name / f'{db_name}.db'
    # plot_study(db_file)

    work_dir = Path.home() / 'scratch/edgeml/work'
    optuna_dirs = work_dir.glob('optuna_la*')
    for path in optuna_dirs:
        if not path.is_dir():
            continue
        # if 'sws256' in path.as_posix() or 'sws512' in path.as_posix():
        #     continue
        db_file = path / f'{path.name}.db'
        print(path.as_posix(), db_file.as_posix())
        plot_study(db_file, save=True, block=False)
