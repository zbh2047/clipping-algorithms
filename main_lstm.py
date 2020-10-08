import argparse
import time
import math
import numpy as np
import torch
import os
import hashlib
import datetime


###############################################################################
# Data loading code
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

###############################################################################
# Make batch
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

###############################################################################
# Training code
###############################################################################

def evaluate(data_source, test_logger, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    ret = total_loss.item() / len(data_source)
    if test_logger is not None:
        test_logger.log({'epoch': epoch, 'loss': ret, 'ppl': math.exp(ret)})
    return ret

def train(train_logger):
    # Turn on training mode which enables dropout.
    total_loss = 0
    avg_loss = 0
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        if 'gamma' in optimizer.param_groups[0]:
            gamma2 = optimizer.param_groups[0]['gamma']
            optimizer.param_groups[0]['gamma'] = gamma2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        # hidden = nn.Parameter(hidden)

        optimizer.zero_grad()
        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

        loss = raw_loss
        # Activation Regularization
        if args.alpha: loss = loss + sum(
            args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        optimizer.step()

        total_loss += raw_loss.data
        avg_loss += raw_loss.data.item()
        optimizer.param_groups[0]['lr'] = lr2
        if 'gamma' in optimizer.param_groups[0]:
            optimizer.param_groups[0]['gamma'] = gamma2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                  'loss {:7.4f} | ppl {:9.3f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                              elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()

        batch += 1
        i += seq_len

    train_logger.log({'epoch': epoch, 'loss': avg_loss / batch, 'ppl': math.exp(avg_loss / batch)})


if __name__ == '__main__':
    # Run commands
    # python main.py --batch_size 20 --data data/penn --dropouti 0.4
    #       --dropouth 0.25 --seed 141 --epoch 500 --save PTB.pt

    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank LSTM Language Model')
    parser.add_argument('--data', type=str, default='data/penn/',
                        help='location of the data corpus')
    parser.add_argument('--result-dir', type=str, default='result/')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (LSTM, GRU)')
    parser.add_argument('--emsize', type=int, default=400,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=1150,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=30,
                        help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=10,
                        help='gradient clipping')
    parser.add_argument('--momentum', type=float, default=0.0,
                        help='momentum')
    parser.add_argument('--epochs', type=int, default=200,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=70,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropouth', type=float, default=0.3,
                        help='dropout for rnn layers (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.65,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0.1,
                        help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--wdrop', type=float, default=0.5,
                        help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--nonmono', type=int, default=5,
                        help='random seed')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--save', type=str, default=randomhash + '.pt',
                        help='path to save the final model')
    parser.add_argument('--alpha', type=float, default=2,
                        help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    parser.add_argument('--wd', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')
    parser.add_argument('--algo', type=str, default='sgd',
                        help='optimizer to use (sgd, adam)')
    parser.add_argument('--nu', type=float, default=0.7)
    args = parser.parse_args()
    args.tied = True

    from utils import TableLogger, create_result_dir
    result_dir = create_result_dir(args)
    train_logger = TableLogger(os.path.join(result_dir, 'train.log'), ['epoch', 'loss', 'ppl'])
    test_logger = TableLogger(os.path.join(result_dir, 'test.log'), ['epoch', 'loss', 'ppl'])
    args.save = os.path.join(result_dir, args.save)

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    ###############################################################################
    # Load data
    ###############################################################################
    fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        from dataset.nlp_data import Corpus
        print('Producing dataset...')
        corpus = Corpus(args.data)
        torch.save(corpus, fn)

    eval_batch_size = 10
    test_batch_size = 2
    train_data = batchify(corpus.train, args.batch_size, args)
    val_data = batchify(corpus.valid, eval_batch_size, args)
    test_data = batchify(corpus.test, test_batch_size, args)

    ###############################################################################
    # Build the model
    ###############################################################################

    from model.splitcross import SplitCrossEntropyLoss
    from model.awdlstm import RNNModel

    criterion = None

    ntokens = len(corpus.dictionary)
    print(ntokens)
    model = RNNModel(args.model,
                           ntokens,
                           args.emsize,
                           args.nhid,
                           args.nlayers,
                           args.dropout,
                           args.dropouth,
                           args.dropouti,
                           args.dropoute,
                           args.wdrop,
                           args.tied,
                           )
    if not criterion:
        splits = []
        if ntokens > 500000:
            # One Billion
            # This produces fairly even matrix mults for the buckets:
            # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
            splits = [4200, 35000, 180000]
        elif ntokens > 75000:
            # WikiText-103
            splits = [2800, 20000, 76000]
        print('Using', splits)
        criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
    ###
    if args.cuda:
        print('Putting model into cuda')
        model = model.cuda()
        criterion = criterion.cuda()
    ###
    params = list(model.parameters()) + list(criterion.parameters())
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
    print('Args:', args)
    print('Model total parameters:', total_params)
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if
                       (x.requires_grad == True and x.size()))
    print('Model trainable parameters:', total_params)

    print('+' * 89)
    print(model)
    print('+' * 89)
    lr = args.lr
    best_val_loss = []
    stored_loss = 100000000

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        optimizer = None
        from algorithm import SGDClip, MomClip, MixClip, Algorithm, SGD
        if args.algo == 'sgd':
            optimizer = Algorithm(params, SGD, lr=args.lr, wd=args.wd, momentum=args.momentum)
        if args.algo == 'adam':
            optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
        if args.algo == 'sgd_clip':
            optimizer = Algorithm(params, SGDClip, lr=args.lr, wd=args.wd, gamma=args.gamma, momentum=args.momentum)
        if args.algo == 'mom_clip':
            optimizer = Algorithm(params, MomClip, lr=args.lr, wd=args.wd, gamma=args.gamma, momentum=args.momentum)
        if args.algo == 'mix_clip':
            optimizer = Algorithm(params, MixClip, lr=args.lr, wd=args.wd, gamma=args.gamma, momentum=args.momentum, nu=args.nu)
        # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
        # Loop over epochs.

        begin_avg = False
        ax = {}
        avg_cnt = 0
        for epoch in range(0, args.epochs):
            epoch_start_time = time.time()
            train(train_logger)
            if begin_avg:
                avg_cnt += 1
                for prm in model.parameters():
                    if avg_cnt == 1:
                        ax[prm] = prm.data.clone()
                    else:
                        ax[prm].add_(prm.data.sub(ax[prm]).mul(1 / avg_cnt))
                tmp = {}
                for prm in model.parameters():
                    tmp[prm] = prm.data.clone()
                    if len(ax) > 0:
                        prm.data.copy_(ax[prm])

                val_loss2 = evaluate(val_data, test_logger)
                loss_scalar = val_loss2
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:7.4f} | '
                      'valid ppl {:9.3f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
                print('-' * 89)

                if val_loss2 < stored_loss:
                    model_save(args.save)
                    print('Saving Averaged!')
                    stored_loss = val_loss2

                for prm in model.parameters():
                    prm.data.copy_(tmp[prm])

            else:
                val_loss = evaluate(val_data, test_logger, eval_batch_size)
                print('-' * 89)
                loss_scalar = val_loss
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:7.4f} | '
                      'valid ppl {:9.3f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
                print('-' * 89)

                if val_loss < stored_loss:
                    model_save(args.save)
                    print('Saving model (new best validation)')
                    stored_loss = val_loss

                if not begin_avg and isinstance(optimizer, Algorithm) and len(best_val_loss) > args.nonmono \
                    and val_loss > min(best_val_loss[:-args.nonmono]):
                    print('Starting averaging')
                    begin_avg = True

                best_val_loss.append(val_loss)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    model_load(args.save)

    # Run on test data.
    test_loss = evaluate(test_data, None, test_batch_size)
    print('=' * 89)
    print('| End of training | test loss {:7.4f} | test ppl {:9.3f} | test bpc {:8.3f}'.format(
        test_loss, math.exp(test_loss), test_loss / math.log(2)))
    print('=' * 89)

