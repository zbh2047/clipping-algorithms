from torch.optim import Optimizer
import torch
import math

# Note: for all algorithms with momentum prarameter, dampening is 1-momentum

class AlgorithmBase():
    @staticmethod
    def update(paras, state, group_state, *args, **kwargs):
        raise NotImplementedError

class Algorithm(Optimizer):
    """
        Note that kwargs should include all the parameters, such as lr, momentum, etc.
        Parameters are grouped and gradient normalization is applied group-wise.
    """
    def __init__(self, params, algo, **kwargs):
        super(Algorithm, self).__init__(params, kwargs)
        assert issubclass(algo, AlgorithmBase)
        self.algo = algo

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for id, group in enumerate(self.param_groups):
            ps = [p for p in group['params'] if p.grad is not None]
            if 'wd' in group and group['wd'] != 0:
                for p in ps:
                    p.grad.data.add_(p.data, alpha=group['wd'])
            self.algo.update(ps, self.state, self.state['group' + str(id)], **group)
        return loss

def sum_of_square_grad(grads):
    return sum([p.view(-1).dot(p.view(-1)) for p in grads])

def update_momentum(grad, state, momentum):
    if 'momentum_buffer' not in state:
        state['momentum_buffer'] = torch.zeros_like(grad, device=grad.device)
    buf = state['momentum_buffer']
    buf.mul_(momentum).add_(1 - momentum, grad)
    return buf

class SGD(AlgorithmBase):
    # g = momentum * g + (1 - momentum) * grad
    # x = x - lr * g
    @staticmethod
    def update(paras, state, group_state, lr, momentum=0, **kwargs):
        for p in paras:
            d_p = p.grad.data
            if momentum != 0:
                d_p = update_momentum(d_p, state[p], momentum)
            p.data.add_(-lr, d_p)

class NormalizedSGD(AlgorithmBase):
    # g = momentum * g + (1 - momentum) * grad
    # x = x - lr * g / |g|
    @staticmethod
    def update(paras, state, group_state, lr, momentum=0, eps=1e-6, **kwargs):
        d_ps = []
        for p in paras:
            d_p = p.grad.data
            if momentum != 0:
                d_p = update_momentum(d_p, state[p], momentum)
            d_ps.append(d_p)
        sum = math.sqrt(sum_of_square_grad(d_ps))
        for p, d_p in zip(paras, d_ps):
            p.data.add_(-lr / (sum + eps), d_p)

class SGDClip(AlgorithmBase):
    # g = momentum * g + (1 - momentum) * grad
    # x = x - min(lr, gamma / |g|) * g
    @staticmethod
    def update(paras, state, group_state, lr, gamma, momentum=0, **kwargs):
        d_ps = []
        for p in paras:
            d_p = p.grad.data
            if momentum != 0:
                d_p = update_momentum(d_p, state[p], momentum)
            d_ps.append(d_p)
        sum = math.sqrt(sum_of_square_grad(d_ps))
        for p, d_p in zip(paras, d_ps):
            p.data.add_(-min(lr, gamma / sum), d_p)

class Adagrad(AlgorithmBase):
    # g^2 = g^2 + |grad|^2
    # x = x - lr / g * grad
    @staticmethod
    def update(paras, state, group_state, lr, b0, **kwargs):
        sum = sum_of_square_grad([p.grad.data for p in paras])
        if 'sum_buffer' not in group_state:
            group_state['sum_buffer'] = sum.new_ones(1) * b0 ** 2
        group_state['sum_buffer'].add_(sum)
        for p in paras:
            p.data.add_(-lr / math.sqrt(group_state['sum_buffer']), p.grad.data)


class MixClip(AlgorithmBase):
    # see the paper
    @staticmethod
    def update(paras, state, group_state, lr, gamma, momentum=0.999, nu=0.7, **kwargs):
        d_ps = []
        for p in paras:
            d_p = p.grad.data
            if momentum != 0:
                d_p = update_momentum(d_p, state[p], momentum)
            d_ps.append(d_p)
        sum = math.sqrt(sum_of_square_grad(d_ps))
        sum2 = math.sqrt(sum_of_square_grad([p.grad.data for p in paras]))
        for p, d_p in zip(paras, d_ps):
            p.data.add_(-lr * nu / (1 + sum / gamma * lr), d_p)
            p.data.add_(-lr * (1 - nu) / (1 + sum2 / gamma * lr), p.grad.data)

class MomClip(AlgorithmBase):
    @staticmethod
    def update(paras, state, group_state, lr, gamma, momentum=0.9, **kwargs):
        MixClip.update(paras, state, group_state, lr, gamma, momentum=momentum, nu=1, **kwargs)
