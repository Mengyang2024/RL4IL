import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

def getMolDescriptors(smi):
    mol = Chem.MolFromSmiles(smi)
    res = []
    for nm, fn in Descriptors._descList:
        res.append(fn(mol))
    return res

def smiles_to_MACCS(smi): 
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    fp = np.array(fp, float)
    return fp

def smiles_to_ECFP(smi):
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fp = np.array(fp, float)
    return fp

def pred(smi: str, forward_model: any) -> float:
    X = np.array([smiles_to_ECFP(smi)])
    pred_X = forward_model.predict(X, return_std=True)[0][0]
    std_X = forward_model.predict(X, return_std=True)[1][0]
    return [pred_X, std_X]
