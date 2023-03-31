import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from .. import builder


@HEADS.register_module
class LatentPredictContrastHead(nn.Module):
    """Head for contrastive learning.
    """

    def __init__(self, predictor, T=1.0):
        super(LatentPredictContrastHead, self).__init__()
        self.predictor = builder.build_neck(predictor)
        self.T = T

    def init_weights(self, init_linear='normal'):
        self.predictor.init_weights(init_linear=init_linear)

    def forward(self, input, target, labels, loss_weight, head_idx=0):
        """Forward head.

        Args:
            input (Tensor): NxC input features.
            target (Tensor): MxC target features.
            labels (Tensor): N,

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        pred = self.predictor([input])[0]
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = target
        logits = torch.einsum('nc,mc->nm', [pred_norm, target_norm]) 
        loss = nn.CrossEntropyLoss()(logits / self.T, labels) * (2 * self.T) * loss_weight
        pos_cos_dist = logits.gather(1, labels.view(-1, 1)).mean()
        neg_cos_dist = (logits.sum() - pos_cos_dist * labels.numel()) / (logits.numel() - labels.numel())

        return dict(loss=loss, pos_cos_dist=pos_cos_dist, neg_cos_dist=neg_cos_dist)

    def forward_list(self, input, target_list, labels, loss_weight=1.0, head_idx=0):
        pred = self.predictor([input])[0]
        pred_norm = nn.functional.normalize(pred, dim=1)
        loss_dict = {}
        for i in range(len(target_list)):
            target_norm = target_list[i]
            logits = torch.einsum('nc,mc->nm', [pred_norm, target_norm]) 
            loss = nn.CrossEntropyLoss()(logits / self.T, labels) * (2 * self.T) * loss_weight
            pos_cos_dist = logits.gather(1, labels.view(-1, 1)).mean()
            neg_cos_dist = (logits.sum() - pos_cos_dist * labels.numel()) / (logits.numel() - labels.numel())
            loss_dict['loss_t_{}'.format(i)] = loss
            loss_dict['pos_cos_t_{}'.format(i)] = pos_cos_dist
            loss_dict['neg_cos_t_{}'.format(i)] = neg_cos_dist
        return loss_dict


@HEADS.register_module
class LatentMultiPredictContrastHead(nn.Module):
    """Head for contrastive learning.
    """

    def __init__(self, predictor, T=1.0, head_num=1):
        super(LatentMultiPredictContrastHead, self).__init__()
        self.head_num = head_num
        self.predictor = nn.ModuleList([
            builder.build_neck(predictor) for _ in range(self.head_num)
        ])
        self.T = T

    def init_weights(self, init_linear='normal'):
        for i in range(self.head_num):
            self.predictor[i].init_weights(init_linear=init_linear)

    def forward(self, input, target, labels, loss_weight=1.0, head_idx=0):
        """Forward head.

        Args:
            input (Tensor): NxC input features.
            target (Tensor): MxC target features.
            labels (Tensor): N,

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        pred = self.predictor[head_idx % self.head_num]([input])[0]
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = target
        logits = torch.einsum('nc,mc->nm', [pred_norm, target_norm]) 
        loss = nn.CrossEntropyLoss()(logits / self.T, labels) * (2 * self.T) * loss_weight
        pos_cos_dist = logits.gather(1, labels.view(-1, 1)).mean()
        neg_cos_dist = (logits.sum() - pos_cos_dist * labels.numel()) / (logits.numel() - labels.numel())

        return dict(loss=loss, pos_cos_dist=pos_cos_dist, neg_cos_dist=neg_cos_dist)
    
    def forward_list(self, input, target_list, labels, loss_weight=1.0, head_idx=0):
        pred = self.predictor[head_idx % self.head_num]([input])[0]
        pred_norm = nn.functional.normalize(pred, dim=1)
        loss_dict = {}
        for i in range(len(target_list)):
            target_norm = target_list[i]
            logits = torch.einsum('nc,mc->nm', [pred_norm, target_norm]) 
            loss = nn.CrossEntropyLoss()(logits / self.T, labels) * (2 * self.T) * loss_weight
            pos_cos_dist = logits.gather(1, labels.view(-1, 1)).mean()
            neg_cos_dist = (logits.sum() - pos_cos_dist * labels.numel()) / (logits.numel() - labels.numel())
            loss_dict['loss_t_{}'.format(i)] = loss
            loss_dict['pos_cos_t_{}'.format(i)] = pos_cos_dist
            loss_dict['neg_cos_t_{}'.format(i)] = neg_cos_dist
        return loss_dict


@HEADS.register_module
class LatentPredictHead(nn.Module):
    """Head for contrastive learning.
    """

    def __init__(self, predictor, size_average=True):
        super(LatentPredictHead, self).__init__()
        self.predictor = builder.build_neck(predictor)
        self.size_average = size_average

    def init_weights(self, init_linear='normal'):
        self.predictor.init_weights(init_linear=init_linear)

    def forward(self, input, target):
        """Forward head.

        Args:
            input (Tensor): NxC input features.
            target (Tensor): NxC target features.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        pred = self.predictor([input])[0]
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        loss = -2 * (pred_norm * target_norm).sum()
        if self.size_average:
            loss /= input.size(0)
        return dict(loss=loss)


@HEADS.register_module
class LatentClsHead(nn.Module):
    """Head for contrastive learning.
    """

    def __init__(self, predictor):
        super(LatentClsHead, self).__init__()
        self.predictor = nn.Linear(predictor.in_channels,
                                   predictor.num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def init_weights(self, init_linear='normal'):
        normal_init(self.predictor, std=0.01)

    def forward(self, input, target):
        """Forward head.

        Args:
            input (Tensor): NxC input features.
            target (Tensor): NxC target features.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        pred = self.predictor(input)
        with torch.no_grad():
            label = torch.argmax(self.predictor(target), dim=1).detach()
        loss = self.criterion(pred, label)
        return dict(loss=loss)
