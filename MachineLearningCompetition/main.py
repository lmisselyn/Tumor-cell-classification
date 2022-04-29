import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# https://towardsdatascience.com/feature-selection-techniques-for-classification-and-python-tips-for-their-application-10c0ddd7918b
# feature selection
# https://academic.oup.com/bioinformatics/article/36/5/1360/5585747?login=false
# ML on single-cell expression data

# --------------------------------------------DATA PREPROCESSING ----------------------------------------------
"""
train_df = pd.read_csv("ML-A5-2022_train.csv", index_col=0)
train_df.replace(to_replace='low', value=0, inplace=True, regex=True)
train_df.replace(to_replace='medium', value=1, inplace=True, regex=True)
train_df.replace(to_replace='high', value=2, inplace=True, regex=True)

# numpy array
train_np = train_df.values

columns_name = train_df.columns
index_name = train_df.index
#train_sample1 = train_np[np.random.randint(train_np.shape[0], size=100), :]
#train_sample2 = train_np[np.random.randint(train_np.shape[0], size=100), :]

imputer1 = SimpleImputer(missing_values=np.nan, strategy="median")
imputer2 = KNNImputer(missing_values=np.nan, n_neighbors=2, weights="uniform")
train_imputed_data = imputer2.fit_transform(train_np)
train_transformed_data = pd.DataFrame(train_imputed_data, dtype=float, columns=columns_name, index=index_name)
train_transformed_data.to_csv(path_or_buf="transformed_data.csv")
print(train_imputed_data)
"""

# --------------------------------------------FEATURE SELECTION ----------------------------------------------

train_df = pd.read_csv("knn_imputed_data.csv", index_col=0)
columns_name = train_df.columns
print(columns_name)

X_train = train_df[columns_name[:34979]]
Y_train = train_df['label']

print(Y_train)
print(X_train)
