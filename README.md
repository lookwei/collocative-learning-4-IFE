## Collocative-learning for Immunofixation Electrophoresis (IFE) Analysis 

Immunofixation Electrophoresis (IFE) analysis is of great importance to the diagnosis of Multiple Myeloma, which is among the top-9 cancer killers in the United States, but has rarely been studied in the context of deep learning. Two possible reasons are: 

1) the recognition of IFE patterns is dependent on the co-location of bands that forms a binary relation, different from the unary relation (visual features to label) that deep learning is good at modeling; 

2) deep classification models may perform with high accuracy for IFE recognition but is not able to provide firm evidence (where the co-location patterns are) for its predictions, rendering difficulty for technicians to validate the results. 

We propose to address these issues with collocative learning, in which a collocative tensor has been constructed to transform the binary relations into unary relations that are compatible with conventional deep networks, and a location-label-free method that utilizes the Grad-CAM saliency map for evidence backtracking has been proposed for accurate localization. 

In addition, we have proposed Coached Attention Gates that can regulate the inference of the learning to be more consistent with human logic and thus support the evidence backtracking. The experimental results show that the proposed method has obtained a performance gain over its base model ResNet18 by 741.30% in IoU and also outperformed popular deep networks of DenseNet, CBAM, and Inception-v3.

![framework](https://github.com/lookwei/collocative-learning-4-IFE/blob/main/framework.png)

## Quick tour
This is a pytorch implementation of the collocative learning method proposed in our TMI paper [1].

To run the code, pleasure make sure you have prepared your IFE data following the same structure as follows (you can also refer to the examplar data in this repository):

../imgs        (the IFE images with name pattern XXX)
../
 
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
CAM was built for users to get a better understanding of models. Here is a short snippet illustrating its usage:

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

Or you can use the example script like below to show the CAM.

```
python test.py -index 0
```

## Citation

[1] X. -Y. Wei et al., "Deep Collocative Learning for Immunofixation Electrophoresis Image Analysis," in IEEE Transactions on Medical Imaging, doi: 10.1109/TMI.2021.3068404.

```bibtex
@ARTICLE{9385115,  
    author={X. -Y. {Wei} and Z. -Q. {Yang} and X. -L. {Zhang} and G. {Liao} and A. -L. {Sheng} and S. {Kevin Zhou} and Y. {Wu} and L. {Du}},  
    journal={IEEE Transactions on Medical Imaging},   
    title={Deep Collocative Learning for Immunofixation Electrophoresis Image Analysis},    
    year={2021},  
    volume={},  
    number={},  
    pages={1-1},  
    doi={10.1109/TMI.2021.3068404}}
```
