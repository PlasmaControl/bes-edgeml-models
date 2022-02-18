import time
from typing import Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
from numpy import ndarray

from . import utils


class Run:
    def __init__(self, model, device: torch.device, criterion, optimizer: torch.optim.Optimizer,
                 use_focal_loss: bool = False, use_rnn: bool = False, clip_grad: float = None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.use_focal_loss = use_focal_loss
        self.use_rnn = use_rnn
        self.clip_grad = clip_grad
        self.is_vae_ = type(self.model).__name__.lower().startswith('vae')

    def train(self, data_loader: torch.utils.data.DataLoader, epoch: int, scheduler: Union[Callable, None] = None,
              print_every: int = 100, ) -> float:
        batch_time = utils.MetricMonitor()
        data_time = utils.MetricMonitor()
        losses = utils.MetricMonitor()
        kls = utils.MetricMonitor()
        likelihoods = utils.MetricMonitor()
        mses = utils.MetricMonitor()

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
            if self.is_vae_:
                reconstruction, mu, logvar, sample = self.model(images)

                loss, kl, likelihood = self.criterion(data_in=images,
                                                      reconstruction=reconstruction,
                                                      mu=mu,
                                                      logvar=logvar,
                                                      sample=sample,
                                                      logscale=self.model.logscale)

                mseloss = torch.nn.MSELoss(reduction='none')(images, reconstruction)

                # perform similar reduction to loss
                kl = kl.mean()
                kls.update(kl.item(), batch_size)
                likelihood = likelihood.mean()
                likelihoods.update(likelihood.item(), batch_size)
                mseloss = mseloss.mean()
                mses.update(mseloss.item(), batch_size)
            else:
                y_preds = self.model(images)
                if self.use_rnn:
                    y_preds = y_preds.squeeze()[:, -1]
                loss = self.criterion(y_preds.view(-1), labels.type_as(y_preds))

                if self.use_focal_loss:
                    loss = self._focal_loss(labels, y_preds, loss)

            # perform loss reduction
            loss = loss.mean()

            # record loss
            losses.update(loss.item(), batch_size)

            # backpropagate
            loss.backward()

            # clip gradient to avoid explosion
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

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
                print(f"Epoch: [{epoch + 1}][{batch_idx + 1}/{len(data_loader)}] "
                      f"Batch time: {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                      f"Elapsed {utils.time_since(start, float(batch_idx + 1) / len(data_loader))} "
                      f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                      f"{'kl divergence' + kls.val if self.is_vae_ else ''} "
                      f"{'Reconstruction Loss' + likelihoods.val if self.is_vae_ else ''}")

        if self.is_vae_:
            return losses.avg, kls.avg, likelihoods.avg, mses.avg
        else:
            return losses.avg

    def evaluate(self, data_loader: torch.utils.data.DataLoader, print_every: int = 50) -> Tuple[
        float, ndarray, ndarray]:
        batch_time = utils.MetricMonitor()
        data_time = utils.MetricMonitor()
        losses = utils.MetricMonitor()
        kls = utils.MetricMonitor()
        likelihoods = utils.MetricMonitor()
        mses = utils.MetricMonitor()

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
                if self.is_vae_:
                    y_preds, mu, logvar, sample = self.model(images)

                    loss, kl, likelihood = self.criterion(data_in=images,
                                                          reconstruction=y_preds,
                                                          mu=mu,
                                                          logvar=logvar,
                                                          sample=sample,
                                                          logscale=self.model.logscale)

                    mseloss = torch.nn.MSELoss(reduction='none')(images, y_preds)
                    # perform reduction and update monitor
                    kl = kl.mean()
                    kls.update(kl.item(), batch_size)
                    likelihood = likelihood.mean()
                    likelihoods.update(likelihood.item(), batch_size)
                    mseloss = mseloss.mean()
                    mses.update(mseloss.item(), batch_size)
                else:
                    y_preds = self.model(images)
                    if self.use_rnn:
                        y_preds = y_preds.squeeze()[:, -1]
                    loss = self.criterion(y_preds.view(-1), labels.type_as(y_preds))

                    if self.use_focal_loss:
                        loss = self._focal_loss(labels, y_preds, loss)

            # perform loss reduction
            loss = loss.mean()

            losses.update(loss.item(), batch_size)

            # record accuracy
            preds.append(torch.sigmoid(y_preds).cpu().detach().numpy())
            valid_labels.append(labels.cpu().detach().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # display results
            if (batch_idx + 1) % print_every == 0:
                print(f"Evaluating: [{batch_idx + 1}/{len(data_loader)}] "
                      f"Batch time: {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                      f"Elapsed {utils.time_since(start, float(batch_idx + 1) / len(data_loader))} "
                      f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                      f"{'kl divergence' + kls.val if self.is_vae_ else ''} "
                      f"{'Reconstruction Loss' + likelihoods.val if self.is_vae_ else ''}")

        predictions = np.concatenate(preds)
        targets = np.concatenate(valid_labels)
        if self.is_vae_:
            return losses.avg, kls.avg, likelihoods.avg, mses.avg, predictions, targets
        else:
            return losses.avg, predictions, targets

    def _focal_loss(self, y_true, y_preds, loss, gamma=2):
        probas = torch.sigmoid(y_preds)
        loss = torch.where(y_true >= 0.5, (1.0 - probas) ** gamma * loss, probas ** gamma * loss, )
        return loss
