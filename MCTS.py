"""The monte carlo tree
"""
import numpy as np
from rdkit import Chem
import math
import Pred 
# from multiprocessing import get_context
import RNN
from DataPrep import Dataset
import tensorflow as tf
import pandas as pd
from tensorflow import keras

class Node:
    """
    To define a node in the monte carlo tree.

    :param current_SMILES: the SMILES symbol list of the current node
    :param parent: the parent of the current node
    """
    def __init__(self, current_smiles: list = None, parent: "Node" = None) -> None:
        self.current_smiles = current_smiles
        self.parent = parent
        self.visits = 0
        self.total_reward = 0
        self.children = []

    def ucb1(self) -> float:
        """
        To calculate the UCB score of the current node

        :return: UCB1 score
        """
        if self.parent is None:
            ucb1 = 0
        elif self.visits == 0:
            ucb1 = 0
        else:
            ucb1 = self.total_reward * 0.3 / self.visits + math.sqrt(2 * math.log(self.parent.visits) / self.visits)
        return ucb1
    
class BuildTree:
    def __init__(self, model) -> None:
        self.model = model
    
    def selection(self, input_node: "Node") -> "Node":
        """
        Traverse down the tree based on UCB1 until reaching a leaf node

        :param input_node: the input node
        
        :return: return the selected node
        """
        while True:
            if len(input_node.children) == 0:
                break
            max_ucb1_index = np.argmax([child.ucb1() for child in input_node.children])
            input_node = input_node.children[max_ucb1_index]
        return input_node
    
    def expansion(self, input_node: "Node") -> "Node":
        """
        Create a new child node from the selected leaf node

        :param input_node: the input node to expansion

        :return: a node with expanded childern nodes
        """

        expansion_policy = self.model.predict_next_possible_symbols
        possible_next_symbols = expansion_policy(input_node.current_smiles) 
        for next_symbol in possible_next_symbols:
            next_smiles = input_node.current_smiles + [next_symbol]
            new_node = Node(current_smiles=next_smiles, parent=input_node)
            input_node.children.append(new_node)
        return input_node
    
    def simulation(self, input_node:"Node", stock: list, pred_model: any, lammbda: float) -> dict:
        """
        Simulate a random rollout from the selected leaf node

        :param input_node: the input node to run rollout
        :stock: a stock of existing cations, which will be used to check the novelty of the SMILES
        :file_to_store_cations: a file to save generated cations
        :lammbda: a float value for balancing exploration and exploitation

        :return a dict for nodes with scores
        """

        rollout_policy = self.model.predict_complete_SMILES
        node_SMILES_dict = {}
        for child in input_node.children:
            smi = child.current_smiles.copy()
            node_SMILES_dict[child] = rollout_policy(smi)
        score_dict = reward_score(node_SMILES_dict, stock, pred_model, lammbda=lammbda)
        return score_dict
    
    def backpropagation(self, score_dict: dict) -> "Node":
        """
        Update the scores of the nodes along the path

        :param score_dict: a dict to store reward scores for each node
        """

        for edge_node in score_dict:
            score = score_dict[edge_node]
            current_node = edge_node
            while True:
                current_node.visits += 1
                current_node.total_reward += score
                current_node = current_node.parent
                node = current_node
                if current_node.parent is None:
                    current_node.visits += 1
                    current_node.total_reward += score
                    break
        return node
    
    def search(self, stock: list, 
               pred_model:any,
               lammbda: float,
               num_loops: int,
               root_SMILES: list = ['^']) -> None:
        """
        To run the tree search.

        :param pred_model: the forward prediction model
        :param stock: a stock of generated cations
        :param lammbda: a float value for balancing exploration and exploitation
        :param num_loops: number of loops to run
        :root_SMILES: the SMILES of the root node 
        """

        node = Node(current_smiles=root_SMILES)
        for _ in range(num_loops):
            print('loop_%d'%_)
            node = self.selection(node)
            print('current path = %s'%node.current_smiles)
            node = self.expansion(node)
            # print('current path = %s'%node.children[0].current_smiles)
            score_dict = self.simulation(node, stock=stock, pred_model=pred_model, lammbda=lammbda)
            # print('current path = %s'%node.children[0].current_smiles)
            node = self.backpropagation(score_dict)
            # print('current path = %s'%node.children[0].current_smiles)
        
def reward_score(node_SMILES_dict: dict, stock: list, pred_model: any, lammbda: float) -> (dict, list):
    """
    To calculate the reward scores of a series of node
    according to their generated complete SMILES 

    :param node_SMILES_dict: a series of node with their generated complete SMILES 
    :stock: a stock of existing cations, which will be used to check the novelty of the SMILES
    :pred_model: the forward prediction model
    :lammbda: a float value for balancing exploration and exploitation
    
    :return: a dict to store the calculated reward scores for each input node
    :return: a list of found valid SMILES
    """

    score_dict = {}
    for node in node_SMILES_dict:
        score_dict[node] = 0
        smiles_checking_result = check_smiles(node_SMILES_dict[node], stock)
        if smiles_checking_result != 'NaN':
            pred = Pred.pred(smiles_checking_result, pred_model)
            if pred[0] < 10 and pred[0] > 0:
                score_dict[node] = pred[0] + lammbda * pred[1]
            else:
                score_dict[node] = 0
            with open('stock.csv', 'a') as f:
                print('%s,%f,%f,%f'%(smiles_checking_result, pred[0], pred[1], score_dict[node]), file=f)
    return score_dict

def check_smiles(input_SMILES: str, stock: list) -> str:
    """
    To check if a SMILES available.

    :param input_SMILES: the SMILES to be checked
    :stock: a stock of existing cations, which will be used to check the novelty of the SMILES
    
    :return: if available, return the input SMILES, else retrun "NaN"
    """
    try:
        mol = Chem.MolFromSmiles(input_SMILES)
    except Exception:
        return 'NaN'
    if mol != None:
        for at in mol.GetAtoms():
            if at.GetNumRadicalElectrons() != 0:
                return 'NaN'
            else:
                continue
        canonical_smiles = Chem.MolToSmiles(mol)
        if canonical_smiles in stock:
            return 'NaN'
        else:
            pass
        if canonical_smiles.count('+]') == 1 and canonical_smiles.count('-]') == 0 \
           and canonical_smiles.count('+2]') == 0 and canonical_smiles.count('+3]') == 0 \
           and canonical_smiles.count('-2]') == 0 and canonical_smiles.count('-3]') == 0:
            return canonical_smiles
        else:
            return 'NaN'
    else:
        return 'NaN'


if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    model_0 = RNN.LoadFromFile('data/rnn_model_cation')
    smis_0 = np.array(list(set(list(pd.read_csv('dataset_for_cellulose_solubility_ML_model_water_content_less_1%.csv')['cation']))))
    training_data = Dataset(smis_0, total_symbol_list=model_0.total_symbol_list)
    model_1 = RNN.FineTune(model_0.rnn_model, training_data.X, training_data.Y, num_epochs=50, lr=0.0001)
    ann_model = keras.models.load_model('data/ANN_model_for_solubility')
    BuildTree(model=model_1).search(stock=list(smis_0), pred_model=ann_model, num_loops=10000)
