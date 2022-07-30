from importlib import import_module
import oneflow as flow
from oneflow import nn, optim
from oneflow.optim import Optimizer

def get_optimizer(name, parameters, lr, l2=0):
    if name == 'sgd':
        return flow.optim.SGD(parameters, lr=lr, weight_decay=l2)
    elif name == 'adagrad':
        return flow.optim.Adagrad(parameters, lr=lr, weight_decay=l2)
    elif name == 'adam':
        return flow.optim.Adam(parameters, lr=lr, weight_decay=l2) 
    elif name == 'adamax':
        return flow.optim.Adamax(parameters, lr=lr, weight_decay=l2) 
    elif name == 'adadelta':
        return flow.optim.Adadelta(parameters, lr=lr, weight_decay=l2)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))

# load args config
def load_config(filename):
    try:
        dump = flow.load(filename)
    except BaseException:
        print("[ Fail: model loading failed. ]")
    return dump['config']
