import numpy as np

# Select dataset
dataset = ['A', 'B', 'C']
dataset_id = 0
print(dataset[dataset_id])

# Select model
models = ['fNIRS-T', 'fNIRS-PreT']
models_id = 0
print(models[models_id])


test_acc = []
for tr in range(1, 26):
    path = 'save/' + dataset[dataset_id] + '/KFold/' + models[models_id] + '/' + str(tr)
    test_max_acc = open(path + '/test_max_acc.txt', "r")
    string = test_max_acc.read()
    acc = string.split('best_acc=')[1]
    acc = float(acc)
    test_acc.append(acc)

test_acc = np.array(test_acc)
print('mean = %.2f' % np.mean(test_acc))
print('std = %.2f' % np.std(test_acc))

