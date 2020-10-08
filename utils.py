import numpy as np
import torch
import os

def random_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def get_result_dir(args):
    result_dir = args.result_dir
    mp = {
        'dataset': '',
        'model': '',
        'algo': '',
        'loss': '',
        'epochs': 'epoch',
        'batch_size': 'bs',
        'momentum': 'moment',
        'gpu': None,
        'print_freq': None,
        'seed': '',
        'result_dir': None,
        'resume': None,
        'data': None,
        'dist_url': None,
        'dist_backend': None
    }
    for arg in vars(args):
        if arg in mp and mp[arg] is None:
            continue
        value = getattr(args, arg)
        if type(value) == bool:
            value = 'T' if value else 'F'
        if type(value) == list:
            value = str(value).replace(' ', '')
        name = mp.get(arg, arg)
        result_dir += name + str(value) + '_'
    return result_dir

def create_result_dir(args):
    result_dir = get_result_dir(args)
    id = 0
    while True:
        result_dir_id = result_dir + '_%d'%id
        if not os.path.exists(result_dir_id): break
        id += 1
    os.makedirs(result_dir_id)
    return result_dir_id

class Logger(object):
    def __init__(self, dir):
        self.fp = open(dir, 'w')
    def __del__(self):
        self.fp.close()
    def print(self, *args, **kwargs):
        print(*args, file=self.fp, **kwargs)
        print(*args, **kwargs)

class TableLogger(object):
    def __init__(self, path, header):
        import csv
        self.fp = open(path, 'w')
        self.logger = csv.writer(self.fp, delimiter='\t')
        self.logger.writerow(header)
        self.header = header
    def __del__(self):
        self.fp.close()
    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])
        self.logger.writerow(write_values)
        self.fp.flush()

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

