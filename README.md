# fNIRS-Transformer
## Transformer Model for Functional Near-Infrared Spectroscopy Classification
This work has been accepted for publication in the IEEE Journal of Biomedical and Health Informatics ( https://ieeexplore.ieee.org/document/9670659).

Our other work may also be useful to you：

***A General and Scalable Vision Framework for Functional Near-Infrared Spectroscopy Classification*** ( https://ieeexplore.ieee.org/abstract/document/9828508).

***Rethinking Delayed Hemodynamic Responses for fNIRS Classification*** (https://ieeexplore.ieee.org/document/10311392/, GitHub: https://github.com/wzhlearning/fNIRSNet).

##  1.  Abstract

Functional near-infrared spectroscopy (fNIRS) is a promising neuroimaging technology. The fNIRS classification problem has always been the focus of the brain-computer interface (BCI). Inspired by the success of Transformer based on self-attention mechanism in the fields of natural language processing and computer vision, we propose an fNIRS classification network based on Transformer, named **fNIRS-T**. We explore the spatial-level and channel-level representation of fNIRS signals to improve data utilization and network representation capacity. Besides, a preprocessing module, which consists of one-dimensional average pooling and layer normalization, is designed to replace filtering and baseline correction of data preprocessing. It makes fNIRS-T an end-to-end network, called **fNIRS-PreT**. 


<div align='center'>
<img src = 'fig/model.png'>
</div>


## 2. Datasets and Preprocessing

We conduct experiments on three open-access datasets. For data preprocessing, we follow the original literature and code.

### 2.1.  Dataset A
Paper: [1], [2]

Dataset:  http://bnci-horizon-2020.eu/database/data-sets 


<div align='center'>
<img src = 'fig/Dataset A.png'>
</div>

### 2.2.  Dataset B

This is a hybrid EEG+fNIRS dataset, and we only use fNIRS data.

Paper: [3]

Dataset:  http://doc.ml.tu-berlin.de/hBCI 

Github:  https://github.com/JaeyoungShin/hybrid-BCI 

### 2.3.  Dataset C
Paper: [4]

Dataset: https://doi.org/10.6084/m9.figshare.9783755.v1 

Github: https://github.com/JaeyoungShin/fNIRS-dataset 

### 2.4. Data Segmentation

For Dataset A, B, and C, the total number of trials are 348, 1740, and 2250, respectively.  Data segmentation is crucial for deep neural networks to classify fNIRS signals. For sampling points of the fNIRS channel, ***start*** and ***end*** represent the start and end of the data split, respectively. "**2**" means two chromophores: HbO and HbR.

| Dataset |           (*start*, *end*)           | Size (trial, channel*2, *end-start*) |
| :-----: | :----------------------------------: | :----------------------------------: |
|    A    | (0, 140) for MA; (160, 300) for Rest |           (348, 104, 140)            |
|    B    |              (100, 300)              |           (1740, 72, 200)            |
|    C    |              (20, 276)               |           (2250, 40, 256)            |



## 3. Flooding Level

Inspired by piecewise decaying learning rates, we decay the flooding level at the specified epoch to obtain continuous regularization. 

| Dataset |   Model    |      (epoch, flooding level)      |
| :-----: | :--------: | :-------------------------------: |
|  **A**  |  fNIRS-T   |                 -                 |
|  **A**  | fNIRS-PreT |                 -                 |
|  **B**  |  fNIRS-T   | (1, 0.45), (30, 0.40), (50, 0.35) |
|  **B**  | fNIRS-PreT | (1, 0.40), (30, 0.38), (50, 0.35) |
|  **C**  |  fNIRS-T   | (1, 0.45), (30, 0.40), (50, 0.35) |
|  **C**  | fNIRS-PreT | (1, 0.45), (30, 0.40), (50, 0.35) |



## 4. Code

 ***scripts***:  The  directory contains conversion code for the three datasets. Although converting to *.xls* format increases read time and dataset storage space, it is convenient to visualize fNIRS signals. For fNIRS-PreT, the ***filtering*** and ***baseline correction*** codes (in ***B_mat2xls.m*** and ***C_mat2xls.m***) need to be ***manually*** disabled or annotated.

***KFold_Train.py*** :  K-fold cross-validation to train the models.

***KFold_ACC.py*** :  Calculate the average K-fold cross-validation accuracy.

***LOSO_Train.py*** :  Leave-one-subject-out cross-validation (LOSO-CV) to train the models.

***LOSO_Results.py*** :  Evaluate experimental results.

***LOSO_Split.py*** :  One subject's data is used as the test set, and the rest is used as the training set.

***model.py*** :  Code for fNIRS-T and fNIRS-PreT.

***dataloader.py*** :  Load the specified dataset.

4.1 Extract datasets A, B, and C using the mat files from the scripts folder. You will need to download the BBCI library from the dataset link. For fNIRS-PreT, the ***filtering*** and ***baseline correction*** codes (in ***B_mat2xls.m*** and ***C_mat2xls.m***) need to be ***manually*** disabled or annotated.

4.2 Run ***KFold_Train.py*** to perform KFold-CV training. Run ***KFold_ACC.py*** to obtain results.

4.3 Run ***LOSO_Train.py*** to execute LOSO-CV training. Run ***LOSO_Results.py*** to obtain results.

You need to specify the dataset, model, and dataset path before training.

```
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
```



## 5. Citation

If you found the study useful for you, please consider citing it.
```
@ARTICLE{Wang2022Transformer,  
author  = {Wang, Zenghui and Zhang, Jun and Zhang, Xiaochu and Chen, Peng and Wang, Bing},  
journal = {IEEE Journal of Biomedical and Health Informatics},   
title   = {Transformer Model for Functional Near-Infrared Spectroscopy Classification},   
year    = {2022},  
volume  = {26},  
number  = {6},  
pages   = {2559-2569},  
doi     = {10.1109/JBHI.2022.3140531}}
```

Our other work:
```
@ARTICLE{Wang2022VisionfNIRSFramework,
author  = {Wang, Zenghui and Zhang, Jun and Xia, Yi and Chen, Peng and Wang, Bing},
journal = {IEEE Transactions on Neural Systems and Rehabilitation Engineering}, 
title   = {A General and Scalable Vision Framework for Functional Near-Infrared Spectroscopy Classification}, 
year    = {2022},
volume  = {30},
number  = {},
pages   = {1982-1991},
doi     = {10.1109/TNSRE.2022.3190431}}
```

```
@ARTICLE{wang2023fNIRSNet,
author  = {Wang, Zenghui and Fang, Jihong and Zhang, Jun},
journal = {IEEE Transactions on Neural Systems and Rehabilitation Engineering}, 
title   = {Rethinking Delayed Hemodynamic Responses for fNIRS Classification}, 
year    = {2023},
volume  = {31},
number  = {},
pages   = {4528-4538},
doi     = {10.1109/TNSRE.2023.3330911}}
```


## References

[1] G. Bauernfeind, R. Scherer, G. Pfurtscheller, and C. Neuper, “Single-trial classifification of antagonistic oxyhemoglobin responses during mental arithmetic,” *Med. Biol. Eng. Comput.*, vol. 49, no. 9, pp. 979–984, 2011.

[2] G. Pfurtscheller, G. Bauernfeind, S. C. Wriessnegger, and C. Neuper,“Focal frontal (de) oxyhemoglobin responses during simple arithmetic,” *Int. J. Psychophysiol.*, vol. 76, no. 3, pp. 186–192, 2010.

[3] J. Shin, A. von Luhmann, B. Blankertz, D.-W. Kim, J. Jeong, H.-J. Hwang, and K.-R. Muller, “Open access dataset for EEG+NIRS single-trial classifification,” *IEEE Trans. Neural Syst. Rehabil. Eng.*, vol. 25, no. 10, pp. 1735–1745, 2017.

[4] S. Bak, J. Park, J. Shin, and J. Jeong, “Open-access fNIRS dataset for classifification of unilateral finger-and foot-tapping,” *Electronics*, vol. 8, no. 12, p. 1486, 2019.
