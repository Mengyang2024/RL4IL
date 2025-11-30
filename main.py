import RNN
from DataPrep import Dataset
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import MCTS

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


sigma_results = pd.read_csv('sigma_result.csv')
model_0 = RNN.LoadFromFile('data/rnn_model_cation')
smis_0 = sigma_results['cation']
training_data = Dataset(smis_0, total_symbol_list=model_0.total_symbol_list)


def smiles_to_ECFP(smi):
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fp = np.array(fp, float)
    return fp

X = np.array([smiles_to_ECFP(sim) for sim in sigma_results['cation']])
Y = np.array(sigma_results['sigma_calc'])
Y = Y.reshape(-1, 1)

# kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5))
kernel = RBF()
gpr = GaussianProcessRegressor(kernel=kernel)
gpr.fit(X, Y)

MCTS.BuildTree(model=model_0).search(stock=list(smis_0), pred_model=gpr, lammbda=10.0,num_loops=100000)