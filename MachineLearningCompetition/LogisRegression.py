import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.feature_selection import chi2, SelectKBest


# ------------------------------------------------- HELPER --------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

def metrics(y_true, y_pred):
    tp, tn, fp, fn = 0, 0, 0, 0
    n_p = 0
    n_n = 0
    for i in range(y_true.size):
        if y_true[i] == 1:
            n_p += 1
            if y_pred[i] == 1:
                tp += 1
            else:
                fn += 1
        else:
            n_n += 1
            if y_pred[i] == 0:
                tn += 1
            else:
                fp += 1
    return tp, tn, fp, fn, n_p, n_n


def bcr_compute(tp, tn, fp, fn):
    return 0.5 * (tp / (tp + fn) + tn / (tn + fp))


def P_compute(model, x_test, y_test, bcr_estim):

    y_test_pred = model.predict(x_test)
    test_metrics = metrics(y_test, y_test_pred)
    bcr_test = bcr_compute(test_metrics[0], test_metrics[1], test_metrics[2], test_metrics[3])
    delta = np.abs(bcr_test - bcr_estim)

    n_p = test_metrics[4]
    n_n = test_metrics[5]
    p_1 = test_metrics[0]/n_p
    p_2 = test_metrics[1]/n_n
    sig = 0.5 * np.sqrt(((1 - p_1) * p_1 / n_p) + (p_2 * (1 - p_2) / n_n))
    p = bcr_test - delta * (1 - np.exp(-delta / sig))

    print("P : " + str(p))
    print("bcr : " + str(bcr_test))
    print("bcr_estim :" + str(bcr_estim))
    print("p1 : " + str(p_1))
    print("p2 : " + str(p_2))
    print("m1 : " + str(n_p))
    print("m2 : " + str(n_n))
    print("\n")

    return p, bcr_test

# -------------------------------------------- DATA PREPROCESSING--------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""
# Training data

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

# Testing data

test_df = pd.read_csv("ML-A5-2022_test.csv", index_col=0)
test_df.replace(to_replace='low', value=0, inplace=True, regex=True)
test_df.replace(to_replace='medium', value=1, inplace=True, regex=True)
test_df.replace(to_replace='high', value=2, inplace=True, regex=True)

# numpy array
test_np = test_df.values

test_columns_name = test_df.columns
test_index_name = test_df.index

imputer = KNNImputer(missing_values=np.nan, n_neighbors=2, weights="uniform")
test_imputed_data = imputer.fit_transform(test_np)
test_transformed_data = pd.DataFrame(test_imputed_data, dtype=float, columns=test_columns_name, index=test_index_name)
test_transformed_data.to_csv(path_or_buf="transformed_test_data.csv")
"""

# ------------------------------------------------- DATA-----------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

# Training data
train_df = pd.read_csv("knn_imputed_data.csv", index_col=0)
columns_name = train_df.columns

X_train = train_df[columns_name[:34979]]
Y_train = train_df['label'].replace(to_replace=-1, value=0)
X_train_np = X_train.values
Y_train_np = Y_train.values

# Testing data
test_df = pd.read_csv("transformed_test_data.csv", index_col=0)
X_test = test_df[columns_name[:34979]]
test_index_name = test_df.index

# ------------------------------------------FEATURE SELECTION------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

# Select K-best features with chi2 for hypothesis testing

selector = SelectKBest(chi2, k=200)
selector.fit(X_train, Y_train)
mask = selector.get_support(indices=True)
X_kbest = X_train.iloc[:, mask]
X_kbest_to_predict = X_test.iloc[:, mask]

# Select some genes involves in breast cancer

X_revelant = train_df[['RAD51-AS1', 'BCAR1', 'PTENP1', 'TP53AIP1', 'STK11', 'CHEK2', 'ATM']]
X_revelant_to_predict = test_df[['RAD51-AS1', 'BCAR1', 'PTENP1', 'TP53AIP1', 'STK11', 'CHEK2', 'ATM']]

# Data standardisation

X_concat = np.concatenate((X_kbest, X_revelant), axis=1)
X_concat_to_predict = np.concatenate((X_kbest_to_predict, X_revelant_to_predict), axis=1)

X = preprocessing.StandardScaler().fit_transform(X_concat)
X_to_predict = preprocessing.StandardScaler().fit_transform(X_concat_to_predict)

print("shape of data to train :" + str(X.shape))
print("shape of data to predict :" + str(X_to_predict.shape))

# ------------------------------------------LOGISTIC REGRESSION----------------------------------------------
# -----------------------------------------------------------------------------------------------------------

model = LogisticRegression(
    penalty='l2',
    dual=False,
    tol=0.0001,
    C=1.0,
    fit_intercept=True,
    intercept_scaling=1,
    class_weight='balanced',
    random_state=None,
    solver='lbfgs',
    max_iter=1000,
    multi_class='auto',
    verbose=0,
    warm_start=False,
    n_jobs=None,
    l1_ratio=None)

BCR = 0
P = 0
for i in range(20):

    x_train, x_test, y_train, y_test = train_test_split(X, Y_train_np, test_size=0.25, random_state=i)
    model.fit(x_train, y_train)

    print(model.score(x_test, y_test))
    result = P_compute(model, x_test, y_test, 0.7)

    BCR += result[1]
    P += result[0]

# BCR estimator
BCR_estimator = BCR/20
# P estimator
P_estimator = P/20

print("BCR :" + str(BCR_estimator))
print("P :" + str(P_estimator))

# store predicted labels in csv file
y_predicted = np.reshape(model.predict(X_to_predict), (-1, 1))
y_predicted_df = pd.DataFrame(data=y_predicted, dtype=int, columns=["Prediction"], index=test_index_name)
y_predicted_final = y_predicted_df.replace(to_replace=0, value=-1)
y_predicted_final.to_csv(path_or_buf="y_predicted.csv")



