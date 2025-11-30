"""Load and process dataset for RNN training.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *


class Dataset:
    """
    Load a training dataset and process the loaded dataset. 

    :param dataset: a numpy array contains a list of SMILES strings
    """

    def __init__(self, dataset: np.ndarray, total_symbol_list: list) -> None:
        self.dataset: np.ndarray = dataset
        self.total_symbol_list: list = total_symbol_list
        self.X, self.Y = self.process()
    
    def split_dataset(self) -> list:
        """
        Split all the SMILES strings in the loaded dataset into symbol-constructed lists.
        Reconstruct a total symbol list by removing duplicates

        :return: a list contains all the symbol-constructed lists
        :return: a total symbol list
        """
        splitted_smis = []
        for smi in self.dataset:
            symbols = split_SMILES(smi)
            splitted_smis.append(symbols)
        return splitted_smis
        
    def process(self, max_len: int = 110):
        """
        To translate all the SMILES strings in the loaded dataset into integer numpy arrays.

        :param max_len: the length of the integer numpy arrays to be translated. default = 110

        :return: two integer numpy arrays. The first one, which the translated integer numpy array, 
        will used as features for RNN training and the second array, which is offset by one 
        element compared to the first one, will be used as target for RNN training.
        """
        X = []
        Y = []
        splitted_dataset = self.split_dataset()
        for splitted_smi in splitted_dataset:
            x = []
            for symbol in splitted_smi:
                x.append(self.total_symbol_list.index(symbol))
            y = x[1:]
            X.append(np.pad(x, (0, max_len - len(x)), 'constant', constant_values=(0, 0)))
            Y.append(np.pad(y, (0, max_len - len(y)), 'constant', constant_values=(0, 0)))
        X = np.array(X)
        Y = np.array(Y)
        batch_X = np.array([to_one_hot(_, len(self.total_symbol_list)) for _ in X])
        batch_Y = np.array([to_one_hot(_, len(self.total_symbol_list)) for _ in Y])
        return batch_X, batch_Y
