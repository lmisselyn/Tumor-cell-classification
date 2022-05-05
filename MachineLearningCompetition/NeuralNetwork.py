from tensorflow import keras
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.feature_selection import chi2, VarianceThreshold, SelectKBest
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

train_df = pd.read_csv("knn_imputed_data.csv", index_col=0)
columns_name = train_df.columns
X_train = train_df[columns_name[:34979]]
Y_train = train_df['label'].replace(to_replace=-1, value=0)
X_train_np = X_train.values
Y_train_np = Y_train.values

X_new = SelectKBest(chi2, k=200).fit_transform(X_train_np, Y_train_np)
X_revelant = train_df[['RAD51-AS1', 'BCAR1', 'PTENP1', 'TP53AIP1', 'STK11', 'CHEK2', 'ATM']]
X_revelant_np = X_revelant.values

X_concat = np.concatenate((X_new, X_revelant), axis=1)
print(X_concat.shape)

x_train, x_test, y_train, y_test = train_test_split(X_concat, Y_train_np, test_size=0.2, random_state=5, shuffle=True)


def create_baseline():
    # create model
    model = keras.models.Sequential()
    model.add(Dense(X_concat.shape[1], input_dim=X_concat.shape[1], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=100)

"""
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X_concat, Y_train_np, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
"""
callbacks = [
    keras.callbacks.ReduceLROnPlateau(patience=10),
    keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)]
estimator.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=callbacks, batch_size=200, epochs=1000)
print(estimator.predict(x_test))
print(estimator.score(x_test, y_test))

cm = confusion_matrix(y_test, estimator.predict(x_test))

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
