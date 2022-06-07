import torch
import pandas as pd
import numpy as np
from scipy import signal


def Load_Dataset_A(data_path, model):
    """
    Load Dataset A. Since the dataset does not provide code, we follow the original literature to preprocess the data.

    Args:
        data_path (str): dataset path.
        model (str): 'fNIRS-T' or 'fNIRS-PreT'. fNIRS-T uses preprocessed data, and fNIRS-PreT uses raw data.

    Returns:
        feature : fNIRS signal data.
        label : fNIRS labels.
    """
    assert model in ['fNIRS-T', 'fNIRS-PreT'], ''' The model parameter is 'fNIRS-T' or 'fNIRS-PreT' '''
    feature = []
    label = []
    for sub in range(1, 9):
        if sub >= 4:
            trial_num = 4
        else:
            trial_num = 3
        for i in range(1, trial_num + 1):
            times = i
            path = data_path + '/s' + str(sub) + '/s' + str(sub) + str(times) + '_hb.xls'
            hb = pd.read_excel(path, header=None)
            path = data_path + '/s' + str(sub) + '/s' + str(sub) + str(times) + '_trial.xls'
            trial = pd.read_excel(path, header=None)
            path = data_path + '/s' + str(sub) + '/s' + str(sub) + str(times) + '_y.xls'
            y = pd.read_excel(path, header=None)

            hb = np.array(hb)
            trial = np.array(trial)
            # 1:MA, 2:REST
            y = np.array(y)

            HbO = hb[:, 0:52]
            HbR = hb[:, 52:104]

            # fNIRS-T uses preprocessed data
            if model == 'fNIRS-T':
                b, a = signal.butter(4, 0.018, 'lowpass')
                HbO = signal.filtfilt(b, a, HbO, axis=0)
                HbR = signal.filtfilt(b, a, HbR, axis=0)
                b, a = signal.butter(3, 0.002, 'highpass')
                HbO = signal.filtfilt(b, a, HbO, axis=0)
                HbR = signal.filtfilt(b, a, HbR, axis=0)

            HbO = HbO.transpose((1, 0))
            HbR = HbR.transpose((1, 0))

            MA_tr = []
            REST_tr = []
            for i in range(y.shape[0]):
                if y[i, 0] == 1:
                    MA_tr.append(trial[i, 0])
                else:
                    REST_tr.append(trial[i, 0])

            HbO_MA = []
            HbO_BL = []
            HbR_MA = []
            HbR_BL = []

            for i in range(int(len(MA_tr))):
                tr = MA_tr[i]
                HbO_MA.append(HbO[:, tr: tr + 140])
                HbR_MA.append(HbR[:, tr: tr + 140])
                HbO_BL.append(HbO[:, tr + 160: tr + 300])
                HbR_BL.append(HbR[:, tr + 160: tr + 300])

            # fNIRS channels = 52, sampling points = 140
            HbO_MA = np.array(HbO_MA).reshape((-1, 1, 52, 140))
            HbO_BL = np.array(HbO_BL).reshape((-1, 1, 52, 140))
            HbR_MA = np.array(HbR_MA).reshape((-1, 1, 52, 140))
            HbR_BL = np.array(HbR_BL).reshape((-1, 1, 52, 140))

            HbO_MA = np.concatenate((HbO_MA, HbR_MA), axis=1)
            HbO_BL = np.concatenate((HbO_BL, HbR_BL), axis=1)

            for i in range(HbO_MA.shape[0]):
                feature.append(HbO_MA[i, :, :, :])
                feature.append(HbO_BL[i, :, :, :])

                label.append(0)
                label.append(1)

        print(str(sub) + '  OK')

    feature = np.array(feature)
    label = np.array(label)
    print('feature', feature.shape)
    print('label', label.shape)

    return feature, label


def Load_Dataset_B(data_path, start=100, end=300):
    """
    Load Dataset B

    Args:
        data_path (str): dataset path.
        start (int): start sampling point, default=100.
        end (int): end sampling point, default=300.

    Returns:
        feature : fNIRS signal data.
        label : fNIRS labels.
    """
    feature = []
    label = []
    for sub in range(1, 30):
        name = data_path + '/' + str(sub) + '/' + str(sub) + '_oxy.xls'
        oxy = pd.read_excel(name, header=None, sheet_name=None)
        name = data_path + '/' + str(sub) + '/' + str(sub) + '_deoxy.xls'
        deoxy = pd.read_excel(name, header=None, sheet_name=None)
        name = data_path + '/' + str(sub) + '/' + str(sub) + '_desc.xls'
        desc = pd.read_excel(name, header=None)

        HbO = []
        HbR = []
        for i in range(1, 61):
            name = 'Sheet' + str(i)
            HbO.append(oxy[name].values)
            HbR.append(deoxy[name].values)

        # (60, 350, 36) --> (60, 36, 350)
        HbO = np.array(HbO).transpose((0, 2, 1))
        HbR = np.array(HbR).transpose((0, 2, 1))
        desc = np.array(desc)

        HbO_MA = []
        HbO_BL = []
        HbR_MA = []
        HbR_BL = []
        for i in range(60):
            if desc[i, 0] == 1:
                HbO_MA.append(HbO[i, :, start:end])
                HbR_MA.append(HbR[i, :, start:end])
            elif desc[i, 0] == 2:
                HbO_BL.append(HbO[i, :, start:end])
                HbR_BL.append(HbR[i, :, start:end])

        # (30, 36, 200) --> (30, 1, 36, 200)
        HbO_MA = np.array(HbO_MA).reshape((30, 1, 36, end-start))
        HbO_BL = np.array(HbO_BL).reshape((30, 1, 36, end-start))
        HbR_MA = np.array(HbR_MA).reshape((30, 1, 36, end-start))
        HbR_BL = np.array(HbR_BL).reshape((30, 1, 36, end-start))

        # (30, 2, 36, 200)
        HbO_MA = np.concatenate((HbO_MA, HbR_MA), axis=1)
        HbO_BL = np.concatenate((HbO_BL, HbR_BL), axis=1)

        for i in range(30):
            feature.append(HbO_MA[i, :, :, :])
            feature.append(HbO_BL[i, :, :, :])
            label.append(0)
            label.append(1)

        print(str(sub) + '  OK')

    feature = np.array(feature)
    label = np.array(label)

    print('feature ', feature.shape)
    print('label ', label.shape)

    return feature, label



def Load_Dataset_C(data_path, start=20, end=276):
    """
    Load Dataset C

    Args:
        data_path (str): dataset path.
        start (int): start sampling point, default=20.
        end (int): end sampling point, default=276.

    Returns:
        feature : fNIRS signal data.
        label : fNIRS labels.
    """
    feature = []
    label = []
    for num in range(1, 31):
        name = data_path + '/' + str(num) + '/' + str(num) + '.xls'
        Hb_org = pd.read_excel(name, header=None, sheet_name=None)
        name = data_path + '/' + str(num) + '/' + str(num) + '_desc.xls'
        desc = pd.read_excel(name, header=None)

        Hb = []
        for i in range(1, 76):
            name = 'Sheet' + str(i)
            Hb.append(Hb_org[name].values)

        # (75, 347, 40)
        Hb = np.array(Hb)
        desc = np.array(desc)

        HbO_R = []
        HbO_L = []
        HbO_F = []
        HbR_R = []
        HbR_L = []
        HbR_F = []
        for i in range(75):
            if desc[i, 0] == 1:
                HbO_R.append(Hb[i, start:end, :20])
                HbR_R.append(Hb[i, start:end, 20:])
            elif desc[i, 0] == 2:
                HbO_L.append(Hb[i, start:end, :20])
                HbR_L.append(Hb[i, start:end, 20:])
            elif desc[i, 0] == 3:
                HbO_F.append(Hb[i, start:end, :20])
                HbR_F.append(Hb[i, start:end, 20:])

        # (25, 256, 20) --> (25, 20, 256) --> (25, 1, 20, 256)
        HbO_R = np.array(HbO_R).transpose((0, 2, 1)).reshape((25, 1, 20, end - start))
        HbO_L = np.array(HbO_L).transpose((0, 2, 1)).reshape((25, 1, 20, end - start))
        HbO_F = np.array(HbO_F).transpose((0, 2, 1)).reshape((25, 1, 20, end - start))

        HbR_R = np.array(HbR_R).transpose((0, 2, 1)).reshape((25, 1, 20, end - start))
        HbR_L = np.array(HbR_L).transpose((0, 2, 1)).reshape((25, 1, 20, end - start))
        HbR_F = np.array(HbR_F).transpose((0, 2, 1)).reshape((25, 1, 20, end - start))

        HbO_R = np.concatenate((HbO_R, HbR_R), axis=1)
        HbO_L = np.concatenate((HbO_L, HbR_L), axis=1)
        HbO_F = np.concatenate((HbO_F, HbR_F), axis=1)

        for i in range(25):
            feature.append(HbO_R[i, :, :, :])
            feature.append(HbO_L[i, :, :, :])
            feature.append(HbO_F[i, :, :, :])

            label.append(0)
            label.append(1)
            label.append(2)

        print(str(num) + '  OK')

    feature = np.array(feature)
    label = np.array(label)
    print('feature ', feature.shape)
    print('label ', label.shape)

    return feature, label


class Dataset(torch.utils.data.Dataset):
    """
    Load data for training

    Args:
        feature: input data.
        label: class for input data.
        transform: Z-score normalization is used to accelerate convergence (default:True).
    """
    def __init__(self, feature, label, transform=True):
        self.feature = feature
        self.label = label
        self.transform = transform
        self.feature = torch.tensor(self.feature, dtype=torch.float)
        self.label = torch.tensor(self.label, dtype=torch.float)
        print(self.feature.shape)
        print(self.label.shape)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        # z-score normalization
        if self.transform:
            mean, std = self.feature[item].mean(), self.feature[item].std()
            self.feature[item] = (self.feature[item] - mean) / std

        return self.feature[item], self.label[item]



