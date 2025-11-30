"""Tools for model training and predicting
"""
import numpy as np

def to_one_hot(input_array: np.ndarray, output_dim: int) -> np.ndarray:
    """
    Convert the input integer numpy array to one-hot
    
    :param: input_array: the input numpy array
    :output_dim: the dimentions of the output one-hot array
    :return: an one-hot array
    """
    res = np.zeros((len(input_array), output_dim), dtype=int)
    res[np.arange(len(input_array)), input_array] = 1
    return res

def split_SMILES(smi: str) -> list:
    """
    Spit the input SMILES string into individual symbols.

    :param smi: a SMILES string
    :return: a list of splitted symbols
    """
    symbols = []
    current_group = ""

    for char in smi:
        if char == "[":
            if current_group:
                symbols.append(current_group)
                current_group = ""
            current_group += char
        elif char == "]":
            current_group += char
            symbols.append(current_group)
            current_group = ""
        else:
            if current_group:
                current_group += char
            else:
                symbols.append(char)

    if current_group:
        symbols.append(current_group)
    symbols.insert(0, '^')
    symbols.append('$')
    return symbols

class Translater:
    """
    To translate a symbol list to integer numpy array or one-hot array using the input dictionary
    
    :param: symbol_list: the symbol list to be translated
    :path_to_dictionary: path to a toatl symbol list
    """
    def __init__(self, symbol_list: list, path_to_dictionary: str) -> None:
        self.symbol_list = symbol_list
        with open(path_to_dictionary, 'r') as f:
            self.dictionary = f.readlines()[0].split()
    
    def to_integer_array(self, output_len: int = 110) -> np.ndarray:
        """
        To translate a SMILS string to integer numpy array.

        :param output_len: the length of output array, pad with 0
        :return: the translated integer array
        """
        integer_array = []
        for symbol in self.symbol_list:
            integer_array.append(self.dictionary.index(symbol))
        integer_array = np.pad(integer_array, (0, output_len - len(integer_array)), 'constant', constant_values=(0, 0))
        return integer_array
    
    def to_one_hot(self) -> np.ndarray:
        """
        To translate a SMILS string to one-hot numpy array.

        :return: the translated one-hot array
        """
        integer_array = self.to_integer_array()
        one_hot = to_one_hot(integer_array, len(self.dictionary))
        return one_hot
