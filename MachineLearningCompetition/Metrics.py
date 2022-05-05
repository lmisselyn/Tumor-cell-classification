import numpy as np

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


def P_compute(model, x_train, y_train, x_test, y_test):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_metrics = metrics(y_train, y_train_pred)
    test_metrics = metrics(y_test, y_test_pred)

    bcr_train = bcr_compute(train_metrics[0], train_metrics[1], train_metrics[2], train_metrics[3])
    bcr_test = bcr_compute(test_metrics[0], test_metrics[1], test_metrics[2], train_metrics[3])

    delta = np.abs(bcr_train - bcr_test)

    n_p = test_metrics[4]
    n_n = test_metrics[5]
    p_1 = test_metrics[0]/n_p
    p_2 = test_metrics[1]/n_n
    sig = 0.5 * np.sqrt(((1 - p_1) * p_1 / n_p) + (p_2 * (1 - p_2) / n_n))
    p = bcr_train - delta * (1 - np.exp(-delta / sig))

    print("P : " + str(p))
    print("bcr : " + str(bcr_train))
    print("bcr_estim : " + str(bcr_test))
    print("p1 : " + str(p_1))
    print("p2 : " + str(p_2))
    print("m1 : " + str(n_p))
    print("m2 : " + str(n_n))
    print("\n")

    return p




