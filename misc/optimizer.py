import argparse

import torch


def build_optimizer(config, model):
    if isinstance(config, argparse.Namespace):
        config = vars(config)
    if config['optim'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), config['lr'],
                                    momentum=config['momentum'],
                                    weight_decay=config['weight_decay'])
    elif config['optim'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), config['lr'],
                                     betas=config['betas'],
                                     weight_decay=config['weight_decay'])
    elif config['optim'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), config['lr'],
                                     betas=config['betas'],
                                     weight_decay=config['weight_decay'])
    elif config['optim'] == 'lars':
        raise NotImplementedError('LARS optimizer is not implemented yet')
    elif config['optim'] == 'lamb':
        raise NotImplementedError('LAMB optimizer is not implemented yet')

    return optimizer