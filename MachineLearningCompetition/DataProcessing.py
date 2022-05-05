import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np


train_df = pd.read_csv("ML-A5-2022_train.csv", index_col=0)
train_df.replace(to_replace='low', value=0, inplace=True, regex=True)
train_df.replace(to_replace='medium', value=1, inplace=True, regex=True)
train_df.replace(to_replace='high', value=2, inplace=True, regex=True)

# numpy array
train_np = train_df.values

columns_name = train_df.columns
index_name = train_df.index

imputer = KNNImputer(missing_values=np.nan, n_neighbors=2, weights="uniform")
train_imputed_data = imputer.fit_transform(train_np)
train_transformed_data = pd.DataFrame(train_imputed_data, dtype=float, columns=columns_name, index=index_name)
train_transformed_data.to_csv(path_or_buf="transformed_data.csv")
print(train_imputed_data)
