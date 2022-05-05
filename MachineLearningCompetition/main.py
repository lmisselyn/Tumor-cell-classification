import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, VarianceThreshold
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split, StratifiedKFold
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

# https://towardsdatascience.com/feature-selection-techniques-for-classification-and-python-tips-for-their-application-10c0ddd7918b
# feature selection
# https://academic.oup.com/bioinformatics/article/36/5/1360/5585747?login=false
# ML on single-cell expression data

# --------------------------------------------FEATURE SELECTION ----------------------------------------------

# GENES INVOLVES IN BREAST CANCER : RAD51, BCAR1, PTENP1, TP53AIP1, STK11, CHEK2, ATM,
train_df = pd.read_csv("knn_imputed_data.csv", index_col=0)
columns_name = train_df.columns

X_train = train_df[columns_name[:34979]]
Y_train = train_df['label']
X_train_np = X_train.values
Y_train_np = Y_train.values

print(X_train_np.shape)


"""
sel = VarianceThreshold(threshold=(.5 * (1 - .5)))
X_new = sel.fit_transform(X_train_np)
print(X_new.shape)
x_train, x_test, y_train, y_test = train_test_split(X_new, Y_train_np, test_size=0.2, random_state=0, shuffle=True)
print(x_test)
print(y_test)




#---------------------------------------------------------CHI2 FEATURE SELECTION-----------------------------------------------
f_score = chi2(X_train_np, Y_train_np)

p_values = pd.Series(f_score[1])
p_values.index = X_train.columns
print(p_values.sort_values(ascending=True))

f_values = pd.Series(f_score[0])
f_values.index = X_train.columns
print(f_values.sort_values(ascending=False))


X_train_new = SelectKBest(chi2, k=100).fit_transform(X_train_np, Y_train_np)
print(X_train_new.shape)
"""
#---------------------------------------------------------GENES INVOLVES IN BREAST CANCER----------------------------------------------------

X_train_df = train_df[['RAD51-AS1', 'BCAR1', 'PTENP1', 'TP53AIP1', 'STK11', 'CHEK2', 'ATM']]
Y_train_df = train_df['label']
X = X_train_df.values
Y = Y_train_df.values
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)

#----------------------------------------------------------NEURAL NETWORK--------------------------------------------------------------------


