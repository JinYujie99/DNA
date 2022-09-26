# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

#  import higher

from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches, PJS_loss
from domainbed.optimizers import get_optimizer


def to_minibatch(x, y):
    minibatches = list(zip(x, y))
    return minibatches


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    transforms = {}

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams

    def update(self, x, y, **kwargs):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.predict(x)


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)

class DNA(Algorithm):
    """
    Diversified Neural Averaging(DNA)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DNA, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.MCdropClassifier(
            in_features=self.featurizer.n_outputs,
            num_classes=num_classes,
            bottleneck_dim=self.hparams["bottleneck_dim"],
            dropout_rate=self.hparams["dropout_rate"],
            dropout_type=self.hparams["dropout_type"]
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.train_sample_num = 5
        self.lambda_v = self.hparams["lambda_v"]

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        all_f = self.featurizer(all_x)
        loss_pjs = 0.0
        row_index = torch.arange(0, all_x.size(0))

        probs_y = []
        for i in range(self.train_sample_num):
            pred = self.classifier(all_f)
            prob = F.softmax(pred, dim=1)
            prob_y = prob[row_index, all_y]
            probs_y.append(prob_y.unsqueeze(0))
            loss_pjs += PJS_loss(prob, all_y)

        probs_y = torch.cat(probs_y, dim=0)
        X = torch.sqrt(torch.log(2/(1+probs_y)) + probs_y * torch.log(2*probs_y/(1+probs_y)) + 1e-6)
        loss_v = (X.pow(2).mean(dim=0) - X.mean(dim=0).pow(2)).mean()
        loss_pjs /= self.train_sample_num
        loss = loss_pjs - self.lambda_v * loss_v

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "loss_c": loss_pjs.item(), "loss_v": loss_v.item()}

    def predict(self, x):
        return self.network(x)


