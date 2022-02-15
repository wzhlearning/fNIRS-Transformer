import numpy as np


def Split_Dataset_A(sub, feature, label, channels):
    """
    LOSO-CV for Dataset A

    Args:
        sub: leave one subject out.
        feature: input fNIRS signals.
        label: input fNIRS labels.
        channels: fNIRS channels.

    Returns:
        X_train: training set.
        y_train: labels for training set.
        X_test: test set.
        y_test: labels for test set.
    """
    if sub == 1:
        X_test = feature[: 36]
        y_test = label[: 36]
        X_train = feature[36:]
        y_train = label[36:]
    elif sub == 8:
        X_test = feature[300:]
        y_test = label[300:]
        X_train = feature[:300]
        y_train = label[: 300]
    else:
        start, end = 0, 0
        if sub in [2, 3]:
            start = 36 * (sub - 1)
            end = 36 * sub
        elif sub in [4, 5, 6, 7]:
            start = 108 + 48 * (sub - 4)
            end = 108 + 48 * (sub - 3)

        X_test = feature[start: end]
        y_test = label[start: end]
        feature_set_1 = feature[: start]
        label_set_1 = label[:start]
        feature_set_2 = feature[end:]
        label_set_2 = label[end:]
        X_train = np.append(feature_set_1, feature_set_2, axis=0)
        y_train = np.append(label_set_1, label_set_2, axis=0)

    X_train = X_train.reshape((X_train.shape[0], 2, channels, -1))
    X_test = X_test.reshape((X_test.shape[0], 2, channels, -1))

    return X_train, y_train, X_test, y_test


def Split_Dataset_A_Res(sub, feature, label, channels):
    """
    Split one subject's data to evaluate the results of LOSO-CV on Dataset A.

    Args:
        sub: leave one subject out.
        feature: input fNIRS signals.
        label: input fNIRS labels.
        channels: fNIRS channels.

    Returns:
        X_test: test set.
        y_test: labels for test set.
    """
    if sub == 1:
        X_test = feature[: 36]
        y_test = label[: 36]
    elif sub == 8:
        X_test = feature[300:]
        y_test = label[300:]
    else:
        start, end = 0, 0
        if sub in [2, 3]:
            start = 36 * (sub - 1)
            end = 36 * sub
        elif sub in [4, 5, 6, 7]:
            start = 108 + 48 * (sub - 4)
            end = 108 + 48 * (sub - 3)

        X_test = feature[start: end]
        y_test = label[start: end]

    X_test = X_test.reshape((X_test.shape[0], 2, channels, -1))

    return X_test, y_test



def Split_Dataset_B(sub, feature, label, channels):
    """
    LOSO-CV for Dataset B

    Args:
        sub: leave one subject out.
        feature: input fNIRS signals.
        label: input fNIRS labels.
        channels: fNIRS channels.

    Returns:
        X_train: training set.
        y_train: labels for training set.
        X_test: test set.
        y_test: labels for test set.
    """
    if sub == 1:
        X_test = feature[: 60]
        y_test = label[: 60]
        X_train = feature[60:]
        y_train = label[60:]
    elif sub == 29:
        X_test = feature[60 * 28:]
        y_test = label[60 * 28:]
        X_train = feature[:60 * 28]
        y_train = label[: 60 * 28]
    else:
        X_test = feature[60 * (sub - 1): 60 * sub]
        y_test = label[60 * (sub - 1): 60 * sub]
        feature_set_1 = feature[: 60 * (sub - 1)]
        label_set_1 = label[:60 * (sub - 1)]
        feature_set_2 = feature[60 * sub:]
        label_set_2 = label[60 * sub:]
        X_train = np.append(feature_set_1, feature_set_2, axis=0)
        y_train = np.append(label_set_1, label_set_2, axis=0)

    X_train = X_train.reshape((X_train.shape[0], 2, channels, -1))
    X_test = X_test.reshape((X_test.shape[0], 2, channels, -1))

    return X_train, y_train, X_test, y_test


def Split_Dataset_B_Res(sub, feature, label, channels):
    """
    Split one subject's data to evaluate the results of LOSO-CV on Dataset B.

    Args:
        sub: leave one subject out.
        feature: input fNIRS signals.
        label: input fNIRS labels.
        channels: fNIRS channels.

    Returns:
        X_test: test set.
        y_test: labels for test set.
    """
    if sub == 1:
        X_test = feature[: 60]
        y_test = label[: 60]
    elif sub == 29:
        X_test = feature[60 * 28:]
        y_test = label[60 * 28:]
    else:
        X_test = feature[60 * (sub - 1): 60 * sub]
        y_test = label[60 * (sub - 1): 60 * sub]

    X_test = X_test.reshape((X_test.shape[0], 2, channels, -1))

    return X_test, y_test



def Split_Dataset_C(sub, feature, label, channels):
    """
    LOSO-CV for Dataset A

    Args:
        sub: leave one subject out.
        feature: input fNIRS signals.
        label: input fNIRS labels.
        channels: fNIRS channels.

    Returns:
        X_train: training set.
        y_train: labels for training set.
        X_test: test set.
        y_test: labels for test set.
    """
    if sub == 1:
        X_test = feature[: 75]
        y_test = label[: 75]
        X_train = feature[75:]
        y_train = label[75:]
    elif sub == 30:
        X_test = feature[75 * 29:]
        y_test = label[75 * 29:]
        X_train = feature[:75 * 29]
        y_train = label[: 75 * 29]
    else:
        X_test = feature[75 * (sub - 1): 75 * sub]
        y_test = label[75 * (sub - 1): 75 * sub]
        feature_set_1 = feature[: 75 * (sub - 1)]
        label_set_1 = label[:75 * (sub - 1)]
        feature_set_2 = feature[75 * sub:]
        label_set_2 = label[75 * sub:]
        X_train = np.append(feature_set_1, feature_set_2, axis=0)
        y_train = np.append(label_set_1, label_set_2, axis=0)

    X_train = X_train.reshape((X_train.shape[0], 2, channels, -1))
    X_test = X_test.reshape((X_test.shape[0], 2, channels, -1))

    return X_train, y_train, X_test, y_test


def Split_Dataset_C_Res(sub, feature, label, channels):
    """
    Split one subject's data to evaluate the results of LOSO-CV on Dataset C.

    Args:
        sub: leave one subject out.
        feature: input fNIRS signals.
        label: input fNIRS labels.
        channels: fNIRS channels.

    Returns:
        X_test: test set.
        y_test: labels for test set.
    """
    if sub == 1:
        X_test = feature[: 75]
        y_test = label[: 75]
    elif sub == 30:
        X_test = feature[75 * 29:]
        y_test = label[75 * 29:]
    else:
        X_test = feature[75 * (sub - 1): 75 * sub]
        y_test = label[75 * (sub - 1): 75 * sub]

    X_test = X_test.reshape((X_test.shape[0], 2, channels, -1))

    return X_test, y_test