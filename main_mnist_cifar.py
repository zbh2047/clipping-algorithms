import argparse
import torch
import os
import time
from torch.utils.data import DataLoader
from utils import random_seed, create_result_dir, Logger, TableLogger, AverageMeter

parser = argparse.ArgumentParser(description='Optimization')
parser.add_argument('--dataset', default='cifar10(augment)', type=str)
parser.add_argument('--model', default='vgg', type=str)
parser.add_argument('--algo', default='sgd', type=str)
parser.add_argument('--loss', default='cross_entropy', type=str)
parser.add_argument('--epochs', default='50,80,100', type=str)
parser.add_argument('-b', '--batch-size', default=128, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--gamma', default=1e-5, type=float)
parser.add_argument('--b0', default=0.0, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--result-dir', default='result/', type=str)
parser.add_argument('--seed', default=2020, type=int, help='seed for initializing training. ')

def parse_dataset(args, batch_size):
    """
        args: a string containing dataset type and paras
            mnist([augment])
            cifar10([augment])
    """
    from dataset.dataset import mnist, cifar10, NonAugmentDataLoader
    if 'mnist' in args.lower():
        train_dataset, test_dataset = mnist('augment' in args.lower())
    elif 'cifar10' in args.lower():
        train_dataset, test_dataset = cifar10('augment' in args.lower())
    else:
        raise NotImplementedError
    test_loader = None
    if not 'augment' in args.lower():
        train_loader = NonAugmentDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if test_dataset is not None:
            test_loader = NonAugmentDataLoader(test_dataset, batch_size=65536, shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        if test_dataset is not None:
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    return train_dataset, test_dataset, train_loader, test_loader

def parse_model(args, dataset=None):
    """
        args: a string containing model type and paras
            linear(d, out, [bias], [large_w])
            vgg16([bn], [affine], [large_w])
            resnet(depth, width, [large_w])
        Note: if large_w, the final weight layer will have norm 15 times larger.
    """
    from model.vgg_cifar import vgg16
    from model.resnet_cifar import resnet_cifar
    from model.linear import Linear
    if 'linear' in args.lower():
        para = ''.join(list(filter((lambda c: str.isdigit(c) or c == ','), args)))
        d, out = [int(x) for x in para.split(',')[0:2]]
        bias = 'bias' in args.lower()
        weight = 15 if 'large_w' in args.lower() else 1
        return Linear(d, out=out, bias=bias, weight=weight)
    elif 'vgg' in args.lower():
        bn = 'bn' in args.lower()
        affine = 'affine' in args.lower()
        large_w = 'large_w' in args.lower()
        return vgg16(bn=bn, affine=affine, large_weight=large_w)
    elif 'resnet' in args.lower():
        para = ''.join(list(filter((lambda c: str.isdigit(c) or c == ','), args)))
        depth, width = [int(x) for x in para.split(',')[0:2]]
        large_w = 'large_w' in args.lower()
        return resnet_cifar(depth=depth, width_factor=width, affine=True, large_weight=large_w)
    else:
        raise NotImplementedError

def parse_algo(args, model, **kwargs):
    """
        args: a string containing algorithm type and paras
            sgd
            sgd_clip([layer])
            normalized_sgd([layer])
            adagrad([element], [layer])
            qhm_clip([layer])
        Note: if [layer], grad normalization is applied layerwise.
            Here every submodel with direct parameters is considered a layer.
    """
    from algorithm import Algorithm, SGD, SGDClip, NormalizedSGD, Adagrad, MomClip, MixClip
    if 'layer' in args.lower():
        modules = model.modules()
        net_paras = [m.parameters(recurse=False) for m in modules]
        net_paras = dict([('params', para) for para in net_paras if len(para) > 0])
    else:
        net_paras = model.parameters()
        para = ('wd', 'lr', 'momentum', 'gamma')
    if 'normalized_sgd' in args.lower():
        algo = NormalizedSGD
        para = ('wd', 'lr', 'momentum')
    elif 'sgd_clip' in args.lower():
        algo = SGDClip
        para = ('wd', 'lr', 'momentum', 'gamma')
    elif 'mom_clip' in args.lower():
        algo = MomClip
        para = ('wd', 'lr', 'momentum', 'gamma')
    elif 'sgd' in args.lower():
        algo = SGD
        para = ('wd', 'lr', 'momentum')
    elif 'adagrad' in args.lower():
        algo = Adagrad
        para = ('wd', 'lr', 'b0')
    elif 'mix_clip' in args.lower():
        algo = MixClip
        para = ('wd', 'lr', 'momentum', 'gamma')
    else:
        raise NotImplementedError
    return Algorithm(net_paras, algo, **{key: kwargs[key] for key in para})

def parse_loss(args):
    """
        args: exp, exp_multi, or cross_entropy
        regularization can be incorporated into loss 'exp' or 'exp_multi' using
            exp(1e-2) for example
    """
    from torch.nn.functional import cross_entropy
    from functools import partial
    import re
    def cross_entropy_no_wd(outputs, targets, weights):
        return cross_entropy(outputs, targets)
    def exp_wd(outputs, targets, weights, wd):
        return torch.exp(-outputs.view(-1) * targets).mean() + \
               sum([(torch.exp(wd * w) - 1).sum() + (torch.exp(-wd * w) - 1).sum() for w in weights]) / 2
    def exp_multi_wd(outputs, targets, weights, wd):
        neg_one = -torch.ones_like(outputs, device=outputs.device, dtype=outputs.dtype)
        return torch.exp(-outputs * neg_one.scatter_(1, targets.view(-1, 1), 1)).sum(dim=1).mean() + \
               sum([(torch.exp(wd * w) - 1).sum() + (torch.exp(-wd * w) - 1).sum() for w in weights]) / 2
    if 'exp_multi' in args.lower():
        matchObj = re.match('exp_multi\((.*)\)', args)
        if matchObj is not None:
            wd = float(matchObj.group(1))
        else: wd = 0
        return partial(exp_multi_wd, wd=wd)
    elif 'exp' in args.lower():
        matchObj = re.match('exp_multi\((.*)\)', args)
        if matchObj is not None:
            wd = float(matchObj.group(1))
        else: wd = 0
        return partial(exp_wd, wd=wd)
    elif 'cross_entropy' in args.lower():
        return cross_entropy_no_wd
    else:
        raise NotImplementedError

def parse_epochs(args):
    return [int(epoch) for epoch in args.split(',')]

def adjust_lr(lr, gamma, epochs, epoch, optimizer):
    import bisect
    index = bisect.bisect_right(epochs, epoch)
    lr_now = lr / (10 ** index)
    gamma_row = gamma / (10 ** index)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_now
        if 'gamma' in param_group:
            param_group['gamma'] = gamma_row

def cal_acc(outputs, targets):
    if outputs.size(1) == 1:
        return (outputs.view(-1) * targets > 0).float().mean()
    predicted = torch.max(outputs.data, 1)[1]
    return (predicted == targets).float().mean()

def train(net, loss_fun, epoch, trainloader, optimizer, logger, train_logger, gpu, print_freq):
    logger.print('Epoch %d training start' % (epoch))
    net.train()

    batch_time, data_time, losses, accs = [AverageMeter() for _ in range(4)]
    start = time.time()

    train_loader_len = len(trainloader)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        data_time.update(time.time() - start)
        inputs = inputs.cuda(gpu, non_blocking=True)
        targets = targets.cuda(gpu, non_blocking=True)
        outputs = net(inputs)

        loss = loss_fun(outputs, targets, net.parameters())
        with torch.no_grad():
            losses.update(loss.data.item(), targets.size(0))
            accs.update(cal_acc(outputs.data, targets).mean().item(), targets.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)
        if (batch_idx + 1) % print_freq == 0:
            logger.print('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
                         'Acc {acc.val:.4f} ({acc.avg:.4f})\t'.format(
                epoch, batch_idx + 1, train_loader_len,
                batch_time=batch_time, data_time=data_time, loss=losses, acc=accs))
        start = time.time()

    loss, acc = losses.avg, accs.avg
    train_logger.log({'epoch': epoch, 'loss': loss, 'acc': acc})
    logger.print('Epoch %d:' % (epoch) + 'train' + " loss: " + f'{loss:.6f}' + " acc: " + f'{acc:.4f}')

@torch.no_grad()
def test(net, loss_fun, epoch, testloader, logger, test_logger, gpu, print_freq):
    net.eval()
    batch_time, data_time, losses, accs = [AverageMeter() for _ in range(4)]
    start = time.time()
    with torch.no_grad():
        test_loader_len = len(testloader)
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.cuda(gpu, non_blocking=True)
            targets = targets.cuda(gpu, non_blocking=True)
            outputs = net(inputs)
            loss = loss_fun(outputs, targets, net.parameters())
            losses.update(loss.mean().item(), targets.size(0))
            accs.update(cal_acc(outputs, targets).item(), targets.size(0))
            batch_time.update(time.time() - start)
            start = time.time()
            if (batch_idx + 1) % print_freq == 0:
                logger.print('Test: [{0}/{1}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
                             'Acc {acc.val:.4f} ({acc.avg:.4f})\t'.format(
                    batch_idx + 1, test_loader_len, batch_time=batch_time, loss=losses, acc=accs))

    loss, acc = losses.avg, accs.avg
    test_logger.log({'epoch': epoch, 'loss': loss, 'acc': acc})
    logger.print("Epoch %d: " % (epoch) + 'test' + " loss: " + f'{loss:.6f}' + " acc: " + f'{acc:.4f}')

def main_worker(gpu, args, result_dir):
    torch.backends.cudnn.benchmark = True
    random_seed(args.seed)
    torch.cuda.set_device(gpu)

    train_dataset, test_dataset, train_loader, test_loader = parse_dataset(args.dataset, args.batch_size)
    model = parse_model(args.model, train_dataset)
    model = model.cuda(gpu)
    print('number of prarmeters:', sum([p.numel() for p in model.parameters()]))
    optimizer = parse_algo(args.algo, model, wd=args.wd,
                           lr=args.lr, momentum=args.momentum, gamma=args.gamma, b0=args.b0)
    loss = parse_loss(args.loss)
    epochs = parse_epochs(args.epochs)

    logger = Logger(os.path.join(result_dir, 'log.txt'))
    for arg in vars(args):
        logger.print(arg, '=', getattr(args, arg))
    train_logger = TableLogger(os.path.join(result_dir, 'train.log'), ['epoch', 'loss', 'acc'])
    test_logger = TableLogger(os.path.join(result_dir, 'test.log'), ['epoch', 'loss', 'acc'])

    for epoch in range(0, epochs[-1]):
        adjust_lr(args.lr, args.gamma, epochs, epoch, optimizer)
        train(model, loss, epoch, train_loader, optimizer, logger, train_logger, gpu, args.print_freq)
        if test_dataset is not None:
            test(model, loss, epoch, test_loader, logger, test_logger, gpu, args.print_freq)

def main(father_handle, **extra_argv):
    args = parser.parse_args()
    for key,val in extra_argv.items():
        setattr(args, key, val)
    result_dir = create_result_dir(args)
    if father_handle is not None:
        father_handle.put(result_dir)
    main_worker(args.gpu, args, result_dir)

if __name__ == '__main__':
    main(None)
