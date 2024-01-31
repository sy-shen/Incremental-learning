# Incremental-learning
The project of `Incremental Learning` is to develop strategies to generate stock alpha factors based on deep learning methods. The goal is to learn the additional information of alpha factors beyond a strong factor.
## DataPreprocessing
This part is used for preprocessing the raw data. The preprocessing contains conducting the standardization and constructing sliding windows to make it applicable for neural network inputs. 

## FactorGeneration 
This part is used for generating alpha factors to learn information in addition to a strong factor. Two methods are given and each method utilizes both GRU and ResTCN for testing.
### Method1
Using a neural network to generate factors, combining them equally with strong factors, where the weight of strong factors is learnable. The final factor is obtained with a loss function based on the negative IC of the final factor and the L2 norm of neural network factors.
### Method2
Regressing strong factors from the original label, creating a new label that aims to capture information beyond what strong factors provide. Using this new label, I generated 16 factors through a neural network. These factors were then combined equally with strong factors to derive the final factor.

## FactorEvaluation
This part is used for evaluating the generated factors. The evaluation contains calculating IC, IR, Annualized long excess returns.
