import time
from typing import Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn

import utils


class Run:
    def __init__(
        self,
        model,
        device: torch.device,
        criterion: nn.BCEWithLogitsLoss,
        optimizer: torch.optim.Adam,
        use_focal_loss: bool = False,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.use_focal_loss = use_focal_loss

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
            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)

            # forward pass
            y_preds = self.model(images)
            loss = self.criterion(y_preds.view(-1), labels.type_as(y_preds))

            if self.use_focal_loss:
                loss = self._focal_loss(labels, y_preds, loss)

            # perform loss reduction
            loss = loss.mean()

            # record loss
            losses.update(loss.item(), batch_size)

            # backpropagate
            loss.backward()

            # optimizer step
            self.optimizer.step()

            # elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # step the scheduler if provided
            if scheduler is not None:
                scheduler.step()

            # display results
            if (batch_idx + 1) % print_every == 0:
                print(
                    f"Epoch: [{epoch+1}][{batch_idx+1}/{len(data_loader)}] "
                    f"Batch time: {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                    f"Elapsed {utils.time_since(start, float(batch_idx+1)/len(data_loader))} "
                    f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                )

        return losses.avg

    def evaluate(
        self, data_loader: torch.utils.data.DataLoader, print_every: int = 50
    ) -> Tuple[utils.MetricMonitor, np.ndarray, np.ndarray]:
        batch_time = utils.MetricMonitor()
        data_time = utils.MetricMonitor()
        losses = utils.MetricMonitor()

        # switch the model to evaluation mode
        self.model.eval()
        preds = []
        valid_labels = []
        start = end = time.time()
        for batch_idx, (images, labels) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # send the data to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)

            # compute loss with no backprop
            with torch.no_grad():
                y_preds = self.model(images)

            y_preds = y_preds.view(-1)
            loss = self.criterion(y_preds, labels.type_as(y_preds))

            if self.use_focal_loss:
                loss = self._focal_loss(labels, y_preds, loss)

            # perform loss reduction
            loss = loss.mean()

            losses.update(loss.item(), batch_size)

            # record accuracy
            preds.append(torch.sigmoid(y_preds).cpu().numpy())
            valid_labels.append(labels.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # display results
            if (batch_idx + 1) % print_every == 0:
                print(
                    f"Evaluating: [{batch_idx+1}/{len(data_loader)}] "
                    f"Batch time: {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                    f"Elapsed {utils.time_since(start, float(batch_idx+1)/len(data_loader))} "
                    f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                )
        predictions = np.concatenate(preds)
        targets = np.concatenate(valid_labels)
        return losses.avg, predictions, targets

    def _focal_loss(self, y_true, y_preds, loss, gamma=2):
        probas = torch.sigmoid(y_preds)
        loss = torch.where(
            y_true >= 0.5,
            (1.0 - probas) ** gamma * loss,
            probas ** gamma * loss,
        )
        return loss
