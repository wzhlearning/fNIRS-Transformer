# fNIRS-Transformer
## Transformer Model for Functional Near-Infrared Spectroscopy Classification
This work has been accepted for publication in the IEEE Journal of Biomedical and Health Informatics (J-BHI). *The code will be released here.* 



##  Abstract

Functional near-infrared spectroscopy (fNIRS) is a promising neuroimaging technology. The fNIRS classification problem has always been the focus of the brain-computer interface (BCI). Inspired by the success of Transformer based on self-attention mechanism in the fields of natural language processing and computer vision, we propose an fNIRS classification network based on Transformer, named **fNIRS-T**. We explore the spatial-level and channel-level representation of fNIRS signals to improve data utilization and network representation capacity. Besides, a preprocessing module, which consists of one-dimensional average pooling and layer normalization, is designed to replace filtering and baseline correction of data preprocessing. It makes fNIRS-T an end-to-end network, called **fNIRS-PreT**. 

![fig](fig\model.png)
