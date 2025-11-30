"""RNN model
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils import *
import os
import time


class LoadFromFile:
    """
    Use a pretrained RNN model to predict complete SMILES strings 
    or guess the next symbol from incomplete SMILES symbol list.

    :param path: path to the pretrained RNN model
    """

    def __init__(self, path: str) -> None:
        self.rnn_model = tf.keras.models.load_model(path)
        if path[-1] == '/':
            self.path_to_total_symbol_list = '%stotal_symbol_list'%path
        else:
            self.path_to_total_symbol_list = '%s/total_symbol_list'%path

        with open(self.path_to_total_symbol_list, 'r') as f:
            self.total_symbol_list = f.readlines()[0].split()
    
    def predict_next_possible_symbols(self, symbol_list: list) -> list:
        """
        Guess a list of next possible symbols
        
        :param symbol_list: the incomplete SMILES string symbol list
        :return: next possible symbols
        """
        input = Translater(symbol_list, self.path_to_total_symbol_list).to_one_hot()
        input = np.array([input])
        y = np.asarray(self.rnn_model.predict(input, verbose=None)[0][len(symbol_list)-1]).astype('float64')
        y = y / np.sum(y)
        multinomial_distribution = np.random.multinomial(1, y, 50)
        possible_selection = list(set([np.where(item == 1)[0][0] for item in multinomial_distribution]))
        possible_next_symbols = [self.total_symbol_list[symbol_id] for symbol_id in possible_selection]
        return possible_next_symbols
    
    def predict_next_possible_symbol(self, symbol_list: list) -> str:
        """
        Guess one next possible symbol
        
        :param symbol_list: the incomplete SMILES string symbol list
        :return: next possible symbol
        """

        input = Translater(symbol_list, self.path_to_total_symbol_list).to_one_hot()
        input = np.array([input])
        y = np.asarray(self.rnn_model.predict(input, verbose=None)[0][len(symbol_list)-1]).astype('float64')
        y = y / np.sum(y)
        np.random.seed((os.getpid() * int(time.time())) % 123456789)
        multinomial_distribution = np.random.multinomial(1, y, 1)
        next_symbol = self.total_symbol_list[np.argmax(multinomial_distribution)]
        return next_symbol
    
    def predict_complete_SMILES(self, symbol_list: list, max_atom_num: int = 30) -> str:
        """
        Predict a complete SMILES string

        :param symbol_list: the incomplete SMILES string symbol list
        :param max_atom_num: the max atom number of the generated molecule
        :return: a complete SMILES string
        """
        possible_SMILES_list = symbol_list
        while True:
            next_symbol = self.predict_next_possible_symbol(possible_SMILES_list)
            if next_symbol == '$':
                break
            if len(possible_SMILES_list) > max_atom_num:
                break
            possible_SMILES_list += [next_symbol]
        possible_SMILES = ''.join(possible_SMILES_list[1:])
        return possible_SMILES

    def fine_tune_model(self, X, Y, num_epochs, lr):
        self.rnn_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=lr))
        self.rnn_model.fit(X, Y, epochs=num_epochs)

class FineTune:
    """
    Fine tune a pretrained RNN model to predict complete SMILES strings 
    or guess the next symbol from incomplete SMILES symbol list.

    :param  : pretrained RNN model
    """

    def __init__(self, trained_rnn_model: str, X, Y, num_epochs, lr) -> None:
        colne_model =  keras.models.clone_model(trained_rnn_model)
        colne_model.set_weights(trained_rnn_model.get_weights()) 
        colne_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=lr))
        colne_model.fit(X, Y, epochs=num_epochs)
        self.rnn_model = colne_model
        self.path_to_total_symbol_list = 'data/rnn_model_cation/total_symbol_list'
        with open(self.path_to_total_symbol_list, 'r') as f:
            self.total_symbol_list = f.readlines()[0].split()
    
    def predict_next_possible_symbols(self, symbol_list: list) -> list:
        """
        Guess a list of next possible symbols
        
        :param symbol_list: the incomplete SMILES string symbol list
        :return: next possible symbols
        """
        input = Translater(symbol_list, self.path_to_total_symbol_list).to_one_hot()
        input = np.array([input])
        y = np.asarray(self.rnn_model.predict(input, verbose=None)[0][len(symbol_list)-1]).astype('float64')
        y = y / np.sum(y)
        multinomial_distribution = np.random.multinomial(1, y, 50)
        possible_selection = list(set([np.where(item == 1)[0][0] for item in multinomial_distribution]))
        possible_next_symbols = [self.total_symbol_list[symbol_id] for symbol_id in possible_selection]
        return possible_next_symbols
    
    def predict_next_possible_symbol(self, symbol_list: list) -> str:
        """
        Guess one next possible symbol
        
        :param symbol_list: the incomplete SMILES string symbol list
        :return: next possible symbol
        """

        input = Translater(symbol_list, self.path_to_total_symbol_list).to_one_hot()
        input = np.array([input])
        y = np.asarray(self.rnn_model.predict(input, verbose=None)[0][len(symbol_list)-1]).astype('float64')
        y = y / np.sum(y)
        np.random.seed((os.getpid() * int(time.time())) % 123456789)
        multinomial_distribution = np.random.multinomial(1, y, 1)
        next_symbol = self.total_symbol_list[np.argmax(multinomial_distribution)]
        return next_symbol
    
    def predict_complete_SMILES(self, symbol_list: list, max_atom_num: int = 30) -> str:
        """
        Predict a complete SMILES string

        :param symbol_list: the incomplete SMILES string symbol list
        :param max_atom_num: the max atom number of the generated molecule
        :return: a complete SMILES string
        """
        possible_SMILES_list = symbol_list
        while True:
            next_symbol = self.predict_next_possible_symbol(possible_SMILES_list)
            if next_symbol == '$':
                break
            if len(possible_SMILES_list) > max_atom_num:
                break
            possible_SMILES_list += [next_symbol]
        possible_SMILES = ''.join(possible_SMILES_list[1:])
        return possible_SMILES