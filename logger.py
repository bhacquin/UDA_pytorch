#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 2019

@author: bastien
"""

import pickle
import time
import torch
import matplotlib.pyplot as plt

class Logger:
    def __init__(self,log_folder=None):
        self._memory = {}
        self._saves = 0
        self._maxsize = 1000000
        self._dumps = 0
        self.log_folder = log_folder

    def add_scalar(self, name, data, timestep):
        """
        Saves a scalar
        """
        if isinstance(data, torch.Tensor):
            data = data.item()

        self._memory.setdefault(name, []).append([data, timestep])

        self._saves += 1
        if self._saves == self._maxsize - 1:
            if self.log_folder is None:
                filename = 'log_data_' + str((self._dumps + 1) * self._maxsize) + '.pkl'
            else:
                filename = self.log_folder + '/log_data_' + str((self._dumps + 1) * self._maxsize) + '.pkl'
            with open(filename, 'wb') as output:
                pickle.dump(self._memory, output, pickle.HIGHEST_PROTOCOL)
            self._dumps += 1
            self._saves = 0
            self._memory = {}

    def save(self):
        if self.log_folder is None:
            filename = 'log_data.pkl'
        else:
            filename = self.log_folder + '/log_data.pkl'
        with open(filename, 'wb') as output:
            pickle.dump(self._memory, output, pickle.HIGHEST_PROTOCOL)

    def save_plot(self):
        for key in self._memory.key():
            filename = self.log_folder + str(key)
            data = np.array(self._memory[key])
            plt.plot(data[:, 1], data[:, 0])
            plt.savefig(filename)
            
