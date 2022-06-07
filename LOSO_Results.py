import torch
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score, f1_score
from dataloader import Dataset, Load_Dataset_A, Load_Dataset_B, Load_Dataset_C
from LOSO_Split import Split_Dataset_A_Res, Split_Dataset_B_Res, Split_Dataset_C_Res
from model import fNIRS_T, fNIRS_PreT

if __name__ == "__main__":
    # Select dataset
    dataset = ['A', 'B', 'C']
    dataset_id = 0
    print(dataset[dataset_id])

    # Select model by setting models_id
    models = ['fNIRS-T', 'fNIRS-PreT']
    models_id = 0
    print(models[models_id])

    # Select the specified path
    data_path = 'data'

    # Load dataset, set number of Subjects
    if dataset[dataset_id] == 'A':
        Subjects = 8
        if models[models_id] == 'fNIRS-T':
            feature, label = Load_Dataset_A(data_path, model='fNIRS-T')
        elif models[models_id] == 'fNIRS-PreT':
            feature, label = Load_Dataset_A(data_path, model='fNIRS-PreT')
    elif dataset[dataset_id] == 'B':
        Subjects = 29
        feature, label = Load_Dataset_B(data_path)
    elif dataset[dataset_id] == 'C':
        Subjects = 30
        feature, label = Load_Dataset_C(data_path)

    _, _, channels, sampling_points = feature.shape

    result_acc = []
    result_pre = []
    result_rec = []
    result_f1 = []
    result_kap = []
    for sub in range(1, Subjects+1):
        if dataset[dataset_id] == 'A':
            X_test, y_test = Split_Dataset_A_Res(sub, feature, label, channels)
        elif dataset[dataset_id] == 'B':
             X_test, y_test = Split_Dataset_B_Res(sub, feature, label, channels)
        elif dataset[dataset_id] == 'C':
             X_test, y_test = Split_Dataset_C_Res(sub, feature, label, channels)

        test_set = Dataset(X_test, y_test, transform=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=X_test.shape[0], shuffle=False)
        # -------------------------------------------------------------------------------------------------------------------- #
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if dataset[dataset_id] == 'A':
            if models[models_id] == 'fNIRS-T':
                net = fNIRS_T(n_class=2, sampling_point=sampling_points, dim=64, depth=6, heads=8, mlp_dim=64).to(device)
            elif models[models_id] == 'fNIRS-PreT':
                net = fNIRS_PreT(n_class=2, sampling_point=sampling_points, dim=64, depth=6, heads=8, mlp_dim=64).to(device)
        elif dataset[dataset_id] == 'B':
            if models[models_id] == 'fNIRS-T':
                net = fNIRS_T(n_class=2, sampling_point=sampling_points, dim=64, depth=6, heads=8, mlp_dim=64).to(device)
            elif models[models_id] == 'fNIRS-PreT':
                net = fNIRS_PreT(n_class=2, sampling_point=sampling_points, dim=64, depth=6, heads=8, mlp_dim=64).to(device)
        elif dataset[dataset_id] == 'C':
            if models[models_id] == 'fNIRS-T':
                net = fNIRS_T(n_class=3, sampling_point=sampling_points, dim=128, depth=6, heads=8, mlp_dim=64).to(device)
            elif models[models_id] == 'fNIRS-PreT':
                net = fNIRS_PreT(n_class=3, sampling_point=sampling_points, dim=128, depth=6, heads=8, mlp_dim=64).to(device)

        weight_path = 'save/' + dataset[dataset_id] + '/LOSO/' + models[models_id] + '/' + str(sub) + '/model.pt'
        net.load_state_dict(torch.load(weight_path))
        # -------------------------------------------------------------------------------------------------------------------- #
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                pred = outputs.argmax(dim=1, keepdim=True)

        y_label = labels.cpu()
        y_pred = pred.cpu()

        acc = accuracy_score(y_label, y_pred)

        if dataset[dataset_id] == 'C':
            # Multi-classification using macro mode
            precision = precision_score(y_label, y_pred, average='macro')
            recall = recall_score(y_label, y_pred, average='macro')
            f1 = f1_score(y_label, y_pred, average='macro')
        else:
            precision = precision_score(y_label, y_pred)
            recall = recall_score(y_label, y_pred)
            f1 = f1_score(y_label, y_pred)
        kappa_value = cohen_kappa_score(y_label, y_pred)

        result_acc.append(acc)
        result_pre.append(precision)
        result_rec.append(recall)
        result_f1.append(f1)
        result_kap.append(kappa_value)

        print('\nAccuracy: {:.2f}%'.format(acc * 100))
        print('Precision: {:.2f}%'.format(precision * 100))
        print('Recall: {:.2f}%'.format(recall * 100))
        print("f1-score: %.2f" % (f1 * 100))
        print("kappa: %.2f" % kappa_value)


    result_acc = np.array(result_acc)
    acc_mean, acc_std = float(np.mean(result_acc)), float(np.std(result_acc))
    result_pre = np.array(result_pre)
    pre_mean, pre_std = float(np.mean(result_pre)), float(np.std(result_pre))
    result_rec = np.array(result_rec)
    rec_mean, rec_std = float(np.mean(result_rec)), float(np.std(result_rec))
    result_f1 = np.array(result_f1)
    f1_mean = float(np.mean(result_f1))
    result_kap = np.array(result_kap)
    kap_mean = float(np.mean(result_kap))

    print('\nacc_mean = %.2f, std = %.2f' % (acc_mean * 100, acc_std * 100))
    print('pre_mean = %.2f, std = %.2f' % (pre_mean * 100, pre_std * 100))
    print('rec_mean = %.2f, std = %.2f' % (rec_mean * 100, rec_std * 100))
    print('f1_mean = %.2f' % (f1_mean * 100))
    print('kap_mean = %.2f' % kap_mean)


