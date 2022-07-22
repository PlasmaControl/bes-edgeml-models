import pickle
import sys
import io
import time

import numpy as np
import h5py
import torch
import torchinfo
from sklearn import metrics

try:
    from ..main.train_base import _Trainer
    from ..main.models import Multi_Features_Model
    from ..main.data import ELM_Dataset
except ImportError:
    from bes_models_2.main.train_base import _Trainer
    from bes_models_2.main.models import Multi_Features_Model
    from bes_models_2.main.data import ELM_Dataset

class ELM_Regression_Trainer(_Trainer):

    def __init__(
        self,
        max_elms: int = None,  # limit ELMs
        log_time: bool = False,  # if True, use log(time_to_elm_onset)
        inverse_weight_label: bool = False,  # must be False if log_time is False
        **kwargs,
    ) -> None:

        t_start_init = time.time()

        # init parent class
        super().__init__(**kwargs)

        self._print_kwargs(self.__class__, locals().copy())

        # subclass attributes
        self.max_elms = max_elms
        self.log_time = log_time
        self.inverse_weight_label = inverse_weight_label

        self.regression = True
        if self.regression:
            self.loss_function = torch.nn.MSELoss(reduction="none")
            self.score_function = metrics.r2_score
        else:
            self.loss_function = torch.nn.BCEWithLogitsLoss(reduction="none")
            self.score_function = metrics.f1_score

        self.train_data = None
        self.validation_data = None
        self.test_data = None
        self._get_data()

        if self.test_data_file:
            self._save_test_data()
        
        self.train_dataset = None
        self.validation_dataset = None
        self._make_datasets()

        self.train_data_loader = None
        self.validation_data_loader = None
        self._make_data_loaders()

        self.model = None
        self._make_model()

        self.optimizer = None
        self.scheduler = None
        self._make_optimizer_scheduler_loss()

        self.results = None

        self.logger.info(f"Initialization time {time.time()-t_start_init:.3f} s")

    def train(self):
        best_score = -np.inf
        self.results = {
            'train_loss': np.empty(0),
            'valid_loss': np.empty(0),
            'scores': np.empty(0),
        }

        if not self.regression:
            self.results['roc_scores'] = np.empty(0)

        # send model to device
        self.model = self.model.to(self.device)

        if self.log_time is False and self.inverse_weight_label is True:
            self.inverse_weight_label = False
            self.logger.info("WARNING: setting `inverse_weight_time` to False; required for log_time==False")

        self.logger.info(f"\nBegin training loop with {self.n_epochs} epochs")
        self.logger.info(f"Batches per epoch {len(self.train_data_loader)}")
        t_start_training = time.time()
        # loop over epochs
        for i_epoch in range(self.n_epochs):
            t_start_epoch = time.time()

            self.logger.info(f"\nEp {i_epoch+1:03d}: begin")
            
            train_loss = self._train_epoch()
            if self.regression:
                train_loss = np.sqrt(train_loss)

            self.results['train_loss'] = np.append(
                self.results['train_loss'],
                train_loss,
            )

            valid_loss, predictions, true_labels = self.evaluate()
            if self.regression:
                valid_loss = np.sqrt(valid_loss)

            self.results['valid_loss'] = np.append(
                self.results['valid_loss'],
                valid_loss,
            )

            # apply learning rate scheduler
            self.scheduler.step(valid_loss)

            score = self.score_function(true_labels, predictions)
            self.results['scores'] = np.append(
                self.results['scores'],
                score,
            )

            if not self.regression:
                # ROC-AUC score
                roc_score = metrics.roc_auc_score(
                    true_labels,
                    predictions,
                )
                self.results['roc_scores'] = np.append(
                    self.results['roc_scores'],
                    roc_score,
                )

            # best score and save model
            if score > best_score:
                best_score = score
                self.logger.info(f"Ep {i_epoch+1:03d}: Best score {best_score:.3f}, saving model...")
                # save pytorch checkpoint ...
                # save onnx format ...

            tmp =  f"Ep {i_epoch+1:03d}: "
            tmp += f"train loss {train_loss:.3f}  "
            tmp += f"val loss {valid_loss:.3f}  "
            tmp += f"score {score:.3f}  "
            if not self.regression:
                tmp += f"roc {roc_score:.3f}  "
            tmp += f"ep time {time.time()-t_start_epoch:.1f} s "
            tmp += f"(total time {time.time()-t_start_training:.1f} s)"
            self.logger.info(tmp)

        self.logger.info(f"\nEnd training loop")
        self.logger.info(f"Elapsed time {time.time()-t_start_training:.1f} s")

    def _train_epoch(self):
        # train mode
        self.model.train()
        # accumulate batch-wise losses
        losses = np.array(0)
        # loop over batches
        for i_batch, (signal_windows, labels) in enumerate(self.train_data_loader):
            if (i_batch+1)%self.minibatch_interval == 0:
                t_start_minibatch = time.time()
            # reset gradients
            self.optimizer.zero_grad()
            # send data to device
            signal_windows = signal_windows.to(self.device)
            labels = labels.to(self.device)
            # calc predictions
            predictions = self.model(signal_windows)
            # calc loss
            loss = self.loss_function(
                predictions.squeeze(),
                labels.type_as(predictions),
            )
            if self.inverse_weight_label:
                loss = torch.div(loss, labels)
            # reduce losses
            loss = loss.mean()  # batch loss
            losses = np.append(losses, loss.detach().numpy())  # track batch losses
            # backpropagate
            loss.backward()
            # update model with optimization step
            self.optimizer.step()
            if (i_batch+1)%self.minibatch_interval == 0:
                tmp =  f"  Train batch {i_batch+1:05d}/{len(self.train_data_loader)}  "
                tmp += f"batch loss {loss:.3f} (avg loss {losses.mean():.3f})  "
                tmp += f"minibatch time {time.time()-t_start_minibatch:.3f} s"
                self.logger.info(tmp)
        return losses.mean()  # return avg. batch loss

    def evaluate(self):
        # evaluate mode
        t_start = time.time()
        losses = np.array(0)
        all_predictions = []
        all_labels = []
        for i_batch, (signal_windows, labels) in enumerate(self.validation_data_loader):
            if (i_batch+1)%self.minibatch_interval == 0:
                t_start_minibatch = time.time()
            signal_windows = signal_windows.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                predictions = self.model(signal_windows)
            if not self.regression:
                predictions = predictions.sigmoid()
            loss = self.loss_function(
                predictions.squeeze(),
                labels.type_as(predictions),
            )
            loss = loss.mean()
            if self.inverse_weight_label:
                loss = torch.div(loss, labels)
            losses = np.append(losses, loss.detach().numpy())
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            if (i_batch+1)%self.minibatch_interval==0:
                tmp =  f"  Valid batch {i_batch+1:05d}/{len(self.validation_data_loader)}  "
                tmp += f"batch loss {loss:.3f} (avg loss {losses.mean():.3f})  "
                tmp += f"minibatch time {time.time()-t_start_minibatch:.3f} s"
                self.logger.info(tmp)
        all_labels = np.concatenate(all_labels)
        all_predictions = np.concatenate(all_predictions)
        return losses.mean(), all_predictions, all_labels

    def _make_optimizer_scheduler_loss(self):
        if self.optimizer_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_type.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay,
                momentum=self.sgd_momentum,
                dampening=self.sgd_dampening,
            )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            verbose=True,
        )

    def _make_model(self):
        self.model = Multi_Features_Model(logger=self.logger)
        self.model = self.model.to(self.device)

        self.logger.info("MODEL SUMMARY")

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        input_size = (
            self.batch_size,
            1,
            self.signal_window_size,
            8,
            8,
        )
        x = torch.rand(*input_size)
        x = x.to(self.device)
        tmp_io = io.StringIO()
        sys.stdout = tmp_io
        print()
        torchinfo.summary(self.model, input_size=input_size, device=self.device)
        sys.stdout = sys.__stdout__
        self.logger.info(tmp_io.getvalue())
        self.logger.info(f"Model contains {n_params} trainable parameters")
        self.logger.info(f'Batched input size: {x.shape}')
        self.logger.info(f"Batched output size: {self.model(x).shape}")


    def _make_data_loaders(self):
        self.train_data_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        self.validation_data_loader = torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
    )

    def _make_datasets(self):
        self.train_dataset = ELM_Dataset(
            *self.train_data[0:4], 
            self.signal_window_size,
        )
        self.validation_dataset = ELM_Dataset(
            *self.validation_data[0:4], 
            self.signal_window_size,
        )

    def _save_test_data(self):
        test_data_file = self.output_dir / self.test_data_file
        self.logger.info(f"Test data file: {test_data_file}")
        with test_data_file.open('wb') as f:
            pickle.dump(
                {
                    "signals": self.test_data[0],
                    "labels": self.test_data[1],
                    "sample_indices": self.test_data[2],
                    "window_start": self.test_data[3],
                    "elm_indices": self.test_data[4],
                },
                f,
            )
        self.logger.info(f"  File size: {test_data_file.stat().st_size/1e6:.1f} MB")

    def _get_data(self):

        self.input_data_file = self.input_data_file.resolve()
        assert self.input_data_file.exists(), f"{self.input_data_file} does not exist"
        self.logger.info(f"Data file: {self.input_data_file}")

        with h5py.File(self.input_data_file, "r") as data_file:
            elm_indices = np.array(
                [int(key) for key in data_file], 
                dtype=np.int32,
            )
            time_frames = sum([data_file[key]['time'].shape[0] for key in data_file])
        self.logger.info(f"Events in data file: {elm_indices.size}")
        self.logger.info(f"Total time frames: {time_frames}")

        np.random.shuffle(elm_indices)
        if self.max_elms:
            elm_indices = elm_indices[:self.max_elms]
            self.logger.info(f"Limiting data to {self.max_elms} ELM events")
        n_test_elms = int(self.fraction_test * elm_indices.size)
        n_validation_elms = int(self.fraction_validation * elm_indices.size)
        test_elms, validation_elms, training_elms = np.split(
            elm_indices,
            [n_test_elms, n_test_elms+n_validation_elms]
        )

        self.logger.info(f"Training ELM events: {training_elms.size}")
        self.train_data = self._preprocess_data(
            training_elms,
            shuffle_indices=True,
            oversample_active_elm=True,
        )

        self.logger.info(f"Validation ELM events: {validation_elms.size}")
        self.validation_data = self._preprocess_data(
            validation_elms,
            shuffle_indices=False,
            oversample_active_elm=False,
        )

        self.logger.info(f"Test ELM events: {test_elms.size}")
        self.test_data = self._preprocess_data(
            test_elms,
            shuffle_indices=False,
            oversample_active_elm=False,
        )

    def _get_valid_indices(
        self,
        labels = None,
        signals = None,
    ):
        #### Begin _get_valid_indices ####
        # indices for active elm times in each elm event
        active_elm_indices = np.nonzero(labels == 1)[0]
        active_elm_start_index = active_elm_indices[0]
        # concat on axis 0 (time dimension)
        valid_t0 = np.ones(active_elm_start_index-1, dtype=np.int32)
        valid_t0[-self.signal_window_size + 1:] = 0
        labels = np.arange(active_elm_start_index, 1, -1, dtype=float)
        signals = signals[:active_elm_start_index-1, :, :]

        if self.log_time:
            if np.any(labels == 0):
                assert False
            labels = np.log10(labels, dtype=float)
        #### End _get_valid_indices ####
        return labels, signals, valid_t0

    def _preprocess_data(
        self,
        elm_indices,
        shuffle_indices: bool = False,
        oversample_active_elm: bool = False,
    ) -> None:
        packaged_signals = None
        packaged_window_start = None
        packaged_valid_t0 = []
        packaged_labels = []
        with h5py.File(self.input_data_file, 'r') as h5_file:
            for elm_index in elm_indices:
                elm_key = f"{elm_index:05d}"
                elm_event = h5_file[elm_key]
                signals = np.array(elm_event["signals"], dtype=np.float32)
                # transpose so time dim. first
                signals = np.transpose(signals, (1, 0)).reshape(-1, 8, 8)
                try:
                    labels = np.array(elm_event["labels"], dtype=np.float32)
                except KeyError:
                    labels = np.array(elm_event["manual_labels"], dtype=np.float32)
                labels, signals, valid_t0 = self._get_valid_indices(labels, signals)
                if packaged_signals is None:
                    packaged_window_start = np.array([0])
                    packaged_valid_t0 = valid_t0
                    packaged_signals = signals
                    packaged_labels = labels
                else:
                    last_index = packaged_labels.size - 1
                    packaged_window_start = np.append(
                        packaged_window_start, 
                        last_index + 1
                    )
                    packaged_valid_t0 = np.concatenate([packaged_valid_t0, valid_t0])
                    packaged_signals = np.concatenate([packaged_signals, signals], axis=0)
                    packaged_labels = np.concatenate([packaged_labels, labels], axis=0)                

        # valid indices for data sampling
        packaged_valid_t0_indices = np.arange(packaged_valid_t0.size, dtype="int")
        packaged_valid_t0_indices = packaged_valid_t0_indices[packaged_valid_t0 == 1]

        if shuffle_indices:
            np.random.shuffle(packaged_valid_t0_indices)

        self.logger.info( "  Data tensors -> signals, labels, sample_indices, window_start_indices:")
        for tensor in [
            packaged_signals,
            packaged_labels,
            packaged_valid_t0_indices,
            packaged_window_start,
        ]:
            tmp = f"  shape {tensor.shape}, dtype {tensor.dtype},"
            tmp += f" min {np.min(tensor):.3f}, max {np.max(tensor):.3f}"
            if hasattr(tensor, "device"):
                tmp += f" device {tensor.device[-5:]}"
            self.logger.info(tmp)

        return (
            packaged_signals, 
            packaged_labels, 
            packaged_valid_t0_indices, 
            packaged_window_start, 
            elm_indices,
        )



if __name__=='__main__':
    m = ELM_Regression_Trainer(
        batch_size=32, 
        minibatch_interval=50, 
        fraction_validation=0.2,
        log_time=True,
        inverse_weight_label=True,
    )
    m.train()