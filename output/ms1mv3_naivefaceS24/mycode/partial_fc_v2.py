
import math
from typing import Callable

import torch
from torch import distributed
from torch.nn.functional import linear, normalize
import numpy as np
import torch.nn.functional as F

class my_CE(torch.nn.Module):
    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        fp16: bool = False,
    ):
        super(my_CE, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.embedding_size = embedding_size
        self.fp16 = fp16
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes, embedding_size)))

        # margin_loss
        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        weight = self.weight

        with torch.autocast("cuda",enabled=self.fp16):
            norm_embeddings = normalize(embeddings)
            norm_weight_activated = normalize(weight)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        # 对最近邻负代理相似度的统计计算
        # 仅考虑batch内正类中心
        index = torch.where(labels != -1)[0]
        indices = labels[index].view(-1)
        batch_weight = norm_weight_activated[indices]
        # 随机从全部weight中抽取类中心
        # sample_size = 512
        # random_indices = torch.randint(norm_weight_activated.shape[0], (sample_size,))
        # batch_weight = norm_weight_activated[random_indices]
        cos_sim = batch_weight@norm_weight_activated.T

        cos_sim[cos_sim==1] = 0        # 对方差的影响很小，暂时不处理
        mean = cos_sim.mean()
        std = cos_sim.std()

        logits = self.margin_softmax(logits, labels)

        # # 对正代理最小相似度的理论计算
        S = self.margin_softmax.s
        # C = num_classes = torch.tensor(93431)
        # N = C - 1 
        # N = N * torch.exp(0.5*std**2 * S**2 + mean * S)
        # Pmax =  torch.tensor(0.99)
        # cos_theta = torch.log(N/(1/Pmax-1))/S
        # cos_theta = torch.clamp(cos_theta, min=-1, max=1)
        # theta = torch.arccos(cos_theta)
        
        
        loss_weight = 25*torch.sin(std)      # 这个效果不错
        # alpha = 0
        # loss_weight = torch.exp(alpha*std)

        
        index = torch.where(labels != -1)[0]
        logits_positive = logits[index, labels[index].view(-1)].mean()
        # 理论值：myloss = 0.5 * std**2 * S**2 + mean * S - logits_positive * S
        myloss = 0.5 * std**2 * S + mean - logits_positive
        loss = self.cross_entropy(logits, labels)
        # loss = loss + 0.02*myloss
        # loss = loss_weight * loss

        return loss
    
class naive_CE(torch.nn.Module):
    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        fp16: bool = False,
    ):
        super(naive_CE, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.embedding_size = embedding_size
        self.fp16 = fp16
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes, embedding_size)))

        
        # margin_loss
        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        weight = self.weight

        with torch.autocast("cuda",enabled=self.fp16):
            norm_embeddings = normalize(embeddings)
            norm_weight_activated = normalize(weight)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        logits = self.margin_softmax(logits, labels)
        loss = self.cross_entropy(logits, labels)
        return loss
        