# iScience_StiffnessCode
Code for Machine learning for iScience Paper
The jupyter notebook iScience_Stifness_Paper_Code_JN_KP_DI_PJ_RR.ipynb contains the code for the analysis done for the paper.
Create a conda environment and install/ import all the packages listed below;
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
plt.rc("font", size=14)
import numpy as np
import math
from copy import deepcopy
from collections import OrderedDict
import tensorflow as tf
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import load_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle as pk
from sklearn import preprocessing 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import balanced_accuracy_score 
from sklearn.metrics import accuracy_score 
from sklearn.neighbors import KernelDensity 
from scipy import integrate
import scipy.stats
import shap
import seaborn as sns
# sns.set(style="white")
# sns.set(style="whitegrid", color_codes=True)
%matplotlib inline
The data files used in the jupyter notebook are (a)young_modulus_gaussian_cells_final_list.csv (for stiffness analysis),(b)young_modulus_sensitivity_ratios.csv (for the substrate sensitivity analysis) and (c)morphology_frame_red_proteomic_allgenes.txt.tar.gz (for the proteomics and neural network analysis)
The repository also contains the train set (X_train_data_janmey_prot_youngmodulus_complex.csv), test set(X_test_data_janmey_prot_youngmodulus_complex.csv) and the neural network model developed for the stiffness prediction(neural_model_morph_janmey_prot_youngmodulus_complex.hdf5).
