## [Collocative-learning for Immunofixation Electrophoresis (IFE) Analysis](https://doi.org/10.1109/TMI.2021.3068404) 

Immunofixation Electrophoresis (IFE) analysis is of great importance to the diagnosis of Multiple Myeloma, which is among the top-9 cancer killers in the United States, but has rarely been studied in the context of deep learning. Two possible reasons are: 

1) the recognition of IFE patterns is dependent on the co-location of bands that forms a binary relation, different from the unary relation (visual features to label) that deep learning is good at modeling; 

2) deep classification models may perform with high accuracy for IFE recognition but is not able to provide firm evidence (where the co-location patterns are) for its predictions, rendering difficulty for technicians to validate the results. 

We propose to address these issues with collocative learning, in which a collocative tensor has been constructed to transform the binary relations into unary relations that are compatible with conventional deep networks, and a location-label-free method that utilizes the Grad-CAM saliency map for evidence backtracking has been proposed for accurate localization. 

In addition, we have proposed Coached Attention Gates that can regulate the inference of the learning to be more consistent with human logic and thus support the evidence backtracking. The experimental results show that the proposed method has obtained a performance gain over its base model ResNet18 by 741.30% in IoU and also outperformed popular deep networks of DenseNet, CBAM, and Inception-v3.

![framework](https://github.com/lookwei/collocative-learning-4-IFE/blob/main/framework.png)

## Quick tour
This is a pytorch implementation of the collocative learning method proposed in our TMI paper [1].

To run the code, please make sure you have prepared your IFE data following the same structure as follows (you can also refer to the examplar data in this repository):

../imgs        (the IFE images)

../img_detail    (the csv file that contains additional information) 
 
## Preprocessing
To cut IFE images into lanes:

```
from Segmentation import *
csv_path = "../img_detail/detail.csv"
ImageSegmentation = DTWImageSegmentation('../imgs/', '../img_blob/')
ImageSegmentation.segment_resized_G003_img(csv_path = csv_path)
```

## Tensor construction
To construct a collocative tensor:

```
from Collaboration import *
create_similarity_dataset(csv_path = csv_path, save_path = "../sim_data/euc_100.npy")
```

## Start training
To train a model with the prepared dataset:

```
from params import *
from Model_Train import *
train(args)
```

## All together
Alternatively, you can run all these steps together with one line:

```
python train.py
```

## Visualization
To show the visualization result of the IFE (given by the index) as presented in our paper:

```
from Visualization import *
index = 0
model_path = '../final_model/ResNet_fold_0.pkl'
data = np.load("../sim_data/euc_100.npy")*get_mask(100)
csv_path = "../img_detail/detail.csv"
show = Grad_Cam_Main(index=index,
                     data=data[index], 
                     csv_path = csv_path,
                     model_path= model_path)
bind_value = show()
```

Or you can use:

```
python test.py -index 0
```

## Datasets
Due to the privacy issue, we cannot distribute the original IFE dataset used in the paper "Deep Collocative Learning for Immunofixation Electrophoresis Image Analysis". However, we creat a simulated dataset which is with the similar appearance and distributions as the original one. Our human technicians have gone through the dataset to make sure that it resembles the orginal one to the maximun extent. The dataset is at ``../imgs/Simulated_IFE_imgs.zip`` We hope it can help initiate your IFE study and verify your methods. The distribution and performance of our method on this simulated dataset with the comparision to those of the original one are as follows.

Label  | Non-M  | IgG-κ | IgG-λ  | IgA-κ | IgA-λ | IgM-κ  | IgM-λ | κ | λ 
:-----------: |:-----------: |:---------: |:---------: |:---------: |:---------: |:---------: |:---------: |:---------: |:---------: 
Original | 2954  | 435  | 392  | 136 | 198 | 78 | 27 | 37 | 95
Simulated | 3056 | 285| 433 | 148 | 154 | 67 | 33 | 45 | 100

Model  | F1-score (%)
:------------: |:-------------:
Original | 94.20%  
Simulated | 98.38% 

## Citation

[1] X. -Y. Wei, Z. -Q. Yang, X. -L. Zhang, G. Liao, A. -L. Sheng, S. K. Zhou, Y. -K. Wu, L. Du, "Deep Collocative Learning for Immunofixation Electrophoresis Image Analysis," in IEEE Transactions on Medical Imaging, vol. 40, no. 7, pp. 1898-1910, July 2021, https://doi.org/10.1109/TMI.2021.3068404. 

```bibtex
@ARTICLE{9385115,  
author={Wei, Xiao-Yong and Yang, Zhen-Qun and Zhang, Xu-Lu and Liao, Ga and Sheng, Ai-Lin and Zhou, S. Kevin and Wu, Yongkang and Du, Liang},  
journal={IEEE Transactions on Medical Imaging},   
title={Deep Collocative Learning for Immunofixation Electrophoresis Image Analysis},   
year={2021},  
volume={40},  
number={7},  
pages={1898-1910},  
doi={10.1109/TMI.2021.3068404}}
```

## Further reading

The collocative learning has also been used for ECG-base eating monitoring and image retrieval. See papers below if you're interested.

[1] X. -L. Zhang, Z. -Q. Yang; D. -M. Jiang et al., "Cardiac Evidence Mining for Eating Monitoring using Collocative Electrocardiogram Imagining," TechRxiv. Preprint. https://doi.org/10.36227/techrxiv.18093275.v2 

[2] X. -L. Zhang, Z. -Q. Yang, H. Tian et al., "Indicative Image Retrieval: Turning Blackbox Learning into Grey," arXiv. Preprint. https://arxiv.org/abs/2201.11898