import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


# ------------------------------------BCR VISUALIZATION FOR ONE EXAMPLE--------------------------------------
# -----------------------------------------------------------------------------------------------------------

def visualize(x_test, y_test, model):
    cm = confusion_matrix(y_test, model.predict(x_test))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
    plt.show()


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
            if y_pred[i] == -1:
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
    p_1 = test_metrics[0] / n_p
    p_2 = test_metrics[1] / n_n
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

def data_preprocessing(filename):
    print("Data preprssing... This step can take a long time\n")
    df = pd.read_csv(filename, index_col=0)
    df.replace(to_replace='low', value=0, inplace=True, regex=True)
    df.replace(to_replace='medium', value=1, inplace=True, regex=True)
    df.replace(to_replace='high', value=2, inplace=True, regex=True)

    imputer = KNNImputer(missing_values=np.nan, n_neighbors=2, weights="uniform")
    transformed_data = pd.DataFrame(imputer.fit_transform(df.values), dtype=float, columns=df.columns, index=df.index)
    # train_transformed_data.to_csv(path_or_buf="transformed_data.csv")
    return transformed_data


# ------------------------------------------FEATURE SELECTION------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

def feature_selection(X_train_df, Y_train_df, X_test_df):
    print("Feature Selection...\n")
    # Select K-best features with chi2 for hypothesis testing
    selector = SelectKBest(chi2, k=65)
    selector.fit(X_train_df, Y_train_df)
    mask = selector.get_support(indices=True)
    X_kbest = X_train_df.iloc[:, mask]
    X_kbest_to_predict = X_test_df.iloc[:, mask]

    # Select some genes involves in breast cancer
    X_revelant = X_train_df[['RAD51-AS1', 'BCAR1', 'PTENP1', 'TP53AIP1', 'STK11', 'CHEK2', 'ATM']]
    X_revelant_to_predict = X_test_df[['RAD51-AS1', 'BCAR1', 'PTENP1', 'TP53AIP1', 'STK11', 'CHEK2', 'ATM']]

    # Data standardisation

    X = preprocessing.StandardScaler().fit_transform(np.concatenate((X_kbest, X_revelant), axis=1))
    X_to_predict = preprocessing.StandardScaler().fit_transform(
        np.concatenate((X_kbest_to_predict, X_revelant_to_predict), axis=1))

    print("shape of data to train :" + str(X.shape))
    print("shape of data to predict :" + str(X_to_predict.shape))
    return X, X_to_predict


# ------------------------------------------LOGISTIC REGRESSION----------------------------------------------
# -----------------------------------------------------------------------------------------------------------

# -------------------------------------- EXPECTED BCR COMPUTATION -------------------------------------------
# -----------------------------------------------------------------------------------------------------------

def expected_bcr(X_train_np, Y_train_np):
    print("Expected BCR computing ... \n")
    BCR = 0
    P = 0
    for i in range(10):

        x_train, x_test, y_train, y_test = train_test_split(X_train_np, Y_train_np, test_size=0.1, random_state=i)
        model = LogisticRegression(
            penalty='l1',
            dual=False,
            tol=0.0001,
            C=1.0,
            fit_intercept=True,
            class_weight='balanced',
            random_state=None,
            solver='saga',
            max_iter=5000,
            multi_class='ovr', )

        for j in range(10):
            x_train2, x_test2, y_train2, y_test2 = train_test_split(x_train, y_train, test_size=0.166, random_state=j)
            model.fit(x_train2, y_train2)

        result = P_compute(model, x_test, y_test, 0.73)
        BCR += result[1]
        P += result[0]

    # BCR estimator
    BCR_estimator = BCR / 10

    print("BCR_estim :" + str(BCR_estimator))  # final BCR estim = 0.735
    print("P :" + str(P / 10))
    return BCR_estimator


# -------------------------------------- FINAL MODEL TRAINING -------------------------------------------
# -------------------------------------------------------------------------------------------------------

def train_and_predict(X_train_np, Y_train_np, X_to_predict_np, test_index_name=None, store=False):
    print("Final model training and label predicting...\n")
    final_model = LogisticRegression(
        penalty='l1',
        dual=False,
        tol=0.0001,
        C=1.0,
        fit_intercept=True,
        class_weight='balanced',
        random_state=None,
        solver='saga',
        max_iter=5000,
        multi_class='ovr', )

    BCR_verif = 0
    P_verif = 0
    for i in range(20):
        x_train, x_test, y_train, y_test = train_test_split(X_train_np, Y_train_np, test_size=0.1, random_state=i)
        final_model.fit(x_train, y_train)
        final_model.score(x_test, y_test)
        result = P_compute(final_model, x_test, y_test, 0.735)
        BCR_verif += result[1]
        P_verif += result[0]
        if i == 19:
            visualize(x_test, y_test, final_model)

    print("BCR :" + str(BCR_verif / 20))
    print("P :" + str(P_verif / 20))

    # Predicted labels
    y_predicted = np.reshape(final_model.predict(X_to_predict_np), (-1, 1))

    if store:
        y_predicted_df = pd.DataFrame(data=y_predicted, dtype=int, columns=["Prediction"], index=test_index_name)
        # y_predicted_final = y_predicted_df.replace(to_replace=0, value=-1)
        y_predicted_df.to_csv(path_or_buf="y_predic_logregress_final.csv")

    return y_predicted


# ----------------------------------------------- MAIN --------------------------------------------------
# -------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    train_preprocessed_df = data_preprocessing("ML-A5-2022_train.csv")
    test_preprocessed_df = data_preprocessing("ML-A5-2022_test.csv")
    columns_name = train_preprocessed_df.columns
    test_index_name = test_preprocessed_df.index

    # Train data
    X_train_df = train_preprocessed_df[columns_name[:34979]]
    Y_train_df = train_preprocessed_df['label']
    Y_train_np = Y_train_df.values
    # Test data
    X_test_df = test_preprocessed_df[columns_name[:34979]]

    X_train_np, X_to_predict_np = feature_selection(X_train_df, Y_train_df, X_test_df)

    BCR_expected = expected_bcr(X_train_np, Y_train_np)
    print("Expected BCR = " + str(BCR_expected))

    Y_predicted = train_and_predict(X_train_np, Y_train_np, X_to_predict_np, test_index_name, True)
