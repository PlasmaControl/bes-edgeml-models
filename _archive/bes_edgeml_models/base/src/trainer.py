import inspect
import sys
import time
from pathlib import Path
from typing import Tuple, Union, Callable

import numpy as np
import torch

from . import utils


class Run:
    def __init__(
            self,
            model: object,
            device: torch.device,
            criterion: object,
            optimizer: torch.optim.Optimizer,
            use_focal_loss: bool = False,
            use_rnn: bool = False,
            inverse_label_weight: bool = False,
    ):
        """
        Trainer class containing the boilerplate code for training and evaluation.

        Args:
            model (object): Instance of the model being used.
            device (torch.device): Device to run training/inference on.
            criterion (object): Instance of the loss function being used.
            optimizer (torch.optim.Optimizer): Optimizer used during training.
            use_focal_loss (bool): If true, use focal loss. It is supposed to work better in class imbalance problems. See: https://arxiv.org/pdf/1708.02002.pdf
            use_rnn (bool): If true, use a recurrent neural network. It makes sure to take predictions at the last time step.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.use_focal_loss = use_focal_loss
        self.use_rnn = use_rnn
        self.inverse_label_weight = inverse_label_weight

    def train(
            self,
            data_loader: torch.utils.data.DataLoader,
            epoch: int,
            scheduler: Union[Callable, None] = None,
            print_every: int = 100,
    ) -> float:
        batch_time = utils.MetricMonitor()
        data_time = utils.MetricMonitor()
        losses = utils.MetricMonitor()

        # put the model to train mode
        self.model.train()

        start = end = time.time()
        for batch_idx, (images, labels) in enumerate(data_loader):
            # data loading time
            data_time.update(time.time() - end)

            # zero out all the accumulated gradients
            self.optimizer.zero_grad()

            # send the data to device
            images = images.to(self.device, dtype=torch.float)
            labels = labels.to(self.device, dtype=torch.float)

            batch_size = images.size(0)

            # forward pass
            y_preds = self.model(images)

            # use predictions for the last time step for RNN
            if self.use_rnn:
                y_preds = y_preds.squeeze()[:, -1]

            caller = Path(inspect.stack()[1].filename).parent.stem
            if caller == 'turbulence_regime_classification':
                loss = self.criterion(y_preds, labels.type(torch.long))
            elif caller == 'velocimetry':
                loss = self.criterion(y_preds, labels.type_as(y_preds))
            else:
                loss = self.criterion(y_preds.squeeze(), labels.type_as(y_preds))

            if not torch.all(torch.isfinite(loss)):
                assert False

            if self.use_focal_loss:
                loss = self._focal_loss(labels, y_preds, loss)

            if self.inverse_label_weight:
                loss = torch.div(loss, labels)

            # perform loss reduction
            loss = loss.mean()

            # record loss
            losses.update(loss.item(), batch_size)

            # backpropagate
            loss.backward()

            # optimizer step
            self.optimizer.step()

            for p in self.model.parameters():
                if p.requires_grad and not torch.all(torch.isfinite(p)):
                    assert False

            # elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # step the scheduler if provided
            if scheduler is not None:
                scheduler.step()

            # display results
            if (batch_idx + 1) % print_every == 0:
                print(
                    f"Epoch: [{epoch + 1}][{batch_idx + 1}/{len(data_loader)}] "
                    f"Batch time: {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                    f"Elapsed {utils.time_since(start, float(batch_idx + 1) / len(data_loader))} "
                    f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                )

        return losses.avg

    def evaluate(
            self,
            data_loader: torch.utils.data.DataLoader,
            print_every: int = 50,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        batch_time = utils.MetricMonitor()
        data_time = utils.MetricMonitor()
        losses = utils.MetricMonitor()

        # switch the model to evaluation mode
        self.model.eval()
        preds = []
        valid_labels = []
        start = end = time.time()
        images_cwt = None
        for batch_idx, (images, labels) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # send the data to device
            images = images.to(self.device, dtype=torch.float)
            labels = labels.to(self.device, dtype=torch.float)

            batch_size = images.size(0)

            # compute loss with no backprop
            with torch.no_grad():
                y_preds = self.model(images)

            if self.use_rnn:
                y_preds = y_preds.squeeze()[:, -1]

            caller = Path(inspect.stack()[1].filename).parent.stem
            if caller == 'turbulence_regime_classification':
                loss = self.criterion(y_preds, labels.type(torch.long))
            elif caller == 'velocimetry':
                loss = self.criterion(y_preds, labels.type_as(y_preds))
            else:
                loss = self.criterion(y_preds.squeeze(), labels.type_as(y_preds))

            if self.use_focal_loss:
                loss = self._focal_loss(labels, y_preds, loss)

            if self.inverse_label_weight:
                loss = torch.div(loss, labels)

            # perform loss reduction
            loss = loss.mean()

            losses.update(loss.item(), batch_size)

            # record accuracy
            if type(self.criterion).__name__ == 'MSELoss':
                preds.append(y_preds.cpu().numpy())
            elif type(self.criterion).__name__ == 'CrossEntropyLoss':
                preds.append((torch.nn.Softmax(dim=1)(y_preds).cpu().numpy()))
            else:
                preds.append(y_preds.sigmoid().cpu().numpy())
            valid_labels.append(labels.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # display results
            if (batch_idx + 1) % print_every == 0:
                print(
                    f"Evaluating: [{batch_idx + 1}/{len(data_loader)}] "
                    f"Batch time: {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                    f"Elapsed {utils.time_since(start, float(batch_idx + 1) / len(data_loader))} "
                    f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                )
        predictions = np.concatenate(preds)
        targets = np.concatenate(valid_labels)

        return losses.avg, predictions, targets

    @staticmethod
    def _focal_loss(y_true, y_preds, loss, gamma=2):
        probas = torch.sigmoid(y_preds)
        loss = torch.where(
            y_true >= 0.5,
            (1.0 - probas) ** gamma * loss,
            probas ** gamma * loss,
        )
        return loss
