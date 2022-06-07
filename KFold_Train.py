import torch
from sklearn.model_selection import RepeatedKFold
import numpy as np
from model import fNIRS_T, fNIRS_PreT
from dataloader import Dataset, Load_Dataset_A, Load_Dataset_B, Load_Dataset_C
import os


class LabelSmoothing(torch.nn.Module):
    """NLL loss with label smoothing."""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


if __name__ == "__main__":
    # Training epochs
    EPOCH = 120

    # Select dataset
    dataset = ['A', 'B', 'C']
    dataset_id = 0
    print(dataset[dataset_id])

    # Select model
    models = ['fNIRS-T', 'fNIRS-PreT']
    models_id = 0
    print(models[models_id])

    # Select the specified path
    data_path = 'data'

    # Save file and avoid training file overwriting.
    save_path = 'save/' + dataset[dataset_id] + '/KFold/' + models[models_id]
    assert os.path.exists(save_path) is False, 'path is exist'
    os.makedirs(save_path)

    # Load dataset and set flooding levels. Different models may have different flooding levels.
    if dataset[dataset_id] == 'A':
        flooding_level = [0, 0, 0]
        if models[models_id] == 'fNIRS-T':
            feature, label = Load_Dataset_A(data_path, model='fNIRS-T')
        elif models[models_id] == 'fNIRS-PreT':
            feature, label = Load_Dataset_A(data_path, model='fNIRS-PreT')
    elif dataset[dataset_id] == 'B':
        if models[models_id] == 'fNIRS-T':
            flooding_level = [0.45, 0.40, 0.35]
        else:
            flooding_level = [0.40, 0.38, 0.35]
        feature, label = Load_Dataset_B(data_path)
    elif dataset[dataset_id] == 'C':
        flooding_level = [0.45, 0.40, 0.35]
        feature, label = Load_Dataset_C(data_path)

    _, _, channels, sampling_points = feature.shape

    feature = feature.reshape((label.shape[0], -1))
    # 5 × 5-fold-CV
    rkf = RepeatedKFold(n_splits=5, n_repeats=5)
    n_runs = 0
    for train_index, test_index in rkf.split(feature):
        n_runs += 1
        print('======================================\n', n_runs)
        path = save_path + '/' + str(n_runs)
        assert os.path.exists(path) is False, 'sub-path is exist'
        os.makedirs(path)

        X_train = feature[train_index]
        y_train = label[train_index]
        X_test = feature[test_index]
        y_test = label[test_index]

        X_train = X_train.reshape((X_train.shape[0], 2, channels, -1))
        X_test = X_test.reshape((X_test.shape[0], 2, channels, -1))

        train_set = Dataset(X_train, y_train, transform=True)
        test_set = Dataset(X_test, y_test, transform=True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True)

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

        criterion = LabelSmoothing(0.1)
        optimizer = torch.optim.AdamW(net.parameters())
        lrStep = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        # -------------------------------------------------------------------------------------------------------------------- #
        test_max_acc = 0
        for epoch in range(EPOCH):
            net.train()
            train_running_acc = 0
            total = 0
            loss_steps = []
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels.long())

                # Piecewise decay flooding. b is flooding level, b = 0 means no flooding
                if epoch < 30:
                    b = flooding_level[0]
                elif epoch < 50:
                    b = flooding_level[1]
                else:
                    b = flooding_level[2]

                # flooding
                loss = (loss - b).abs() + b

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_steps.append(loss.item())
                total += labels.shape[0]
                pred = outputs.argmax(dim=1, keepdim=True)
                train_running_acc += pred.eq(labels.view_as(pred)).sum().item()

            train_running_loss = float(np.mean(loss_steps))
            train_running_acc = 100 * train_running_acc / total
            print('[%d, %d] Train loss: %0.4f' % (n_runs, epoch, train_running_loss))
            print('[%d, %d] Train acc: %0.3f%%' % (n_runs, epoch, train_running_acc))

            # -------------------------------------------------------------------------------------------------------------------- #
            net.eval()
            test_running_acc = 0
            total = 0
            loss_steps = []
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())

                    loss_steps.append(loss.item())
                    total += labels.shape[0]
                    pred = outputs.argmax(dim=1, keepdim=True)
                    test_running_acc += pred.eq(labels.view_as(pred)).sum().item()

                test_running_acc = 100 * test_running_acc / total
                test_running_loss = float(np.mean(loss_steps))
                print('     [%d, %d] Test loss: %0.4f' % (n_runs, epoch, test_running_loss))
                print('     [%d, %d] Test acc: %0.3f%%' % (n_runs, epoch, test_running_acc))

                if test_running_acc > test_max_acc:
                    test_max_acc = test_running_acc
                    torch.save(net.state_dict(), path + '/model.pt')
                    test_save = open(path + '/test_acc.txt', "w")
                    test_save.write("%.3f" % (test_running_acc))
                    test_save.close()

            lrStep.step()
