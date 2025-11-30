# RL4IL: reinforcement learning model for ionic liquis (ILs)

RL4IL is a repository for inverse design de novo organic ions in ionic liquids utilizing a reinforcement learning (RL) strategy. 

As illustrated in the following Figure, the RL framework comprises three iterative steps. (1) Forward model training: An ML model was developed to predict the ionic conductivity of ILs. (2) Inverse generation: The trained forward model was used as a reward function in an RNN–MCTS inverse generator, enabling de novo design of DCA-based IL cations toward high predicted conductivity. (3) Validation and retraining: Newly generated ILs were validated via MD simulations. The resulting conductivity data were used to expand the training set and improve the forward model’s accuracy. The retrained model was then reintegrated into the RNN–MCTS generator, and the cycle was repeated. 
<img width="612" height="326" alt="image" src="https://github.com/user-attachments/assets/d59fd72d-e720-40b0-b55c-38ce8a0411e4" />


## Usage

### Installation:
Clone the repository and install the required packages:
```shell
conda create -n RL4IL python=3.8
conda activate RL4IL
conda install cudatoolkit=11.8.0
conda install cudnn=8.9.2.26 -c anaconda
pip install tensorflow==2.13.0 rdkit==2023.9.4 pandas==1.5.3 scikit-learn==1.3.0
```

The present repository represents the initial loop of the RL cycle.
### To inverse generate de novo cations:
```shell
python main.py
```
Executing the script generates new cation structures along with their predicted ionic conductivity, which are stored in "stock.csv".
After submitting the high-score candidates for MD simulation-based conductivity evaluation, the resulting data can be appended to the forward-model training set "sigma_results.csv".
This expanded dataset then enables the initiation of the subsequent RL loop.


## Reference

If you find the code useful for your research, please consider citing

Qu, M., Sharma, G., Wada, N. et al. "Inverse Design of Dicyanamide anion Based Ionic Liquids with High Ionic Conductivity", in submitting.

