import torch
import torch.nn as nn
from torch.autograd import Variable

class BackHook(torch.nn.Module):
    def __init__(self, hook):
        super(BackHook, self).__init__()
        self._hook = hook
        self.register_backward_hook(self._backward)

    def forward(self, *inp):
        return inp

    @staticmethod
    def _backward(self, grad_in, grad_out):
        self._hook()
        return None


class WeightDrop(torch.nn.Module):
    """
    Implements drop-connect, as per Merity et al https://arxiv.org/abs/1708.02182
    """
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()
        self.wdrop = BackHook(self._backward)

    def _setup(self):
        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            self.register_parameter(name_w + '_raw', nn.Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self, name_w + '_raw')
            if self.training:
                mask = raw_w.new_ones((raw_w.size(0), 1))
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
                setattr(self, name_w + "_mask", mask)
            else:
                w = raw_w
            rnn_w = getattr(self.module, name_w)
            rnn_w.data.copy_(w)

    def _backward(self):
        # transfer gradients from embeddedRNN to raw params
        for name_w in self.weights:
            raw_w = getattr(self, name_w + '_raw')
            rnn_w = getattr(self.module, name_w)
            raw_w.grad = rnn_w.grad * getattr(self, name_w + "_mask")

    def forward(self, *args):
        self._setweights()
        return self.module(*self.wdrop(*args))

def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = torch.nn.functional.embedding(words, masked_embed_weight,
        padding_idx, embed.max_norm, embed.norm_type,
        embed.scale_grad_by_freq, embed.sparse
    )
    return X

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x