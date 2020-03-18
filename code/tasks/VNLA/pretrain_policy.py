import os
import sys
import numpy as np
import torch
from env import VNLAPretrainBatch
from model import ???

def get_ft_variable(ob_windows):
    '''return a f_t tensor of shape (batch_size, 128by128by4)'''
    pass


class PolicyPretrainer():

    def __init__(self, hparams, model):
        self.load_path = hparams.load_path
        self.save_path = hparams.save_path
        self.model = model

    def load_weights(self):
        pass

    def save_weights(self):
        pass

    def _get_ft_variable(self, ob_windows):
        '''return a f_t tensor of shape (batch_size, 128by128by4)'''
        pass

    def train(self, n_iters):
        pass

    def test(self):
        pass




if __name__ == '__main__':
    # load hparams

    # load model

    # Initialize Pretrainer

    # 
    pass
