# mulit_view_object_detection

## Project Description
Within this project a network architecture was developed to fuse features from multiple images with different viewpoints on the same scene for the purpose of instance segmentation. In order to combine the information from different images, the camera poses are leveraged using projective geometry. By implementing unprojection and projection layers into the network structure, which can transform the information into a 3D coordinate system, the network is not required to learn the rules of projective geometry.
The performance of the developed network is evaluated on the novel InteriorNet dataset. Implementing projective geometry into the network proves to be an applicable approach and the developed modules build a solid base for future work on this topic.

## Foulder Structure
The expected foulder structure is shwon below. The scene folders of InteriorNet are save for example in HD1/train.
The Matterport_Inference Notebook is a demo Notebook to show how inference is performed with a trained model. The Matterport_Interior Notebook is a demo for training.

<pre> 
.
├── data
│   ├── InteriorNet
│   ├── data
│   │   ├── HD1
│   │   │   ├── train
│   │   │   └── val
│   │   └── HD7
│   │       ├── test
│   │       ├── train
│   │       └── val
│   │
│   └── Results
├── logs
├── mask_rcnn_coco.h5
├── mrcnn
│   ├── config.py
│   ├── __init__.py
│   ├── model_multi.py
│   ├── model.py
│   ├── model_transformer.py
│   ├── recurrent.py
│   ├── utils.py
│   └── visualize.py
├── Notebook
│   ├── data_inspection.ipynb
│   ├── Download_InteriorNet.ipynb
│   ├── instances_per_class_in_train.txt
│   ├── instances_per_class_in_val.txt
│   ├── Matterport_Inference.ipynb
│   ├── Matterport_Interior.ipynb
│   ├── Matterport_Interior_transformer.ipynb
│   ├── projection.py
│   └── show_results.ipynb
├── requirements.txt
├── samples
│   ├── demo.ipynb
│   └── interior
</pre>


## Usage
To train a model the file samples/interior/interior_multi.py is used. In this file the necessary settings can be made like the number of views etc. In the folder samples/interior the follwing commands can be used for training and inference.
```
 python interior_multi.py train --dataset ../../data/InteriorNet/data/HD1 --model /path/to/weights --logs ../../logs
 python interior_multi.py evaluate --dataset ../../data/InteriorNet/data/HD1 --model /path/to/weights --logs ../../logs
```
