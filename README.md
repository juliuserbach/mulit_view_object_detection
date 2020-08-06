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
Before training is run the mapping between views has to be created. Run view_mapping.py for the sequential dataset and view_mapping_seq.py for the non-sequential dataset. Both scripts are in samples/interior.

To train a model the file samples/interior/interior_multi.py is used. In this file the necessary settings can be made like the number of views etc. In the folder samples/interior the follwing commands can be used for training and inference on the sequential dataset. To use the non-sequential dataset, the HD7 should be selected.
```
 python interior_multi.py train --dataset ../../data/InteriorNet/data/HD1 --model /path/to/weights --logs ../../logs
 python interior_multi.py evaluate --dataset ../../data/InteriorNet/data/HD1 --model /path/to/weights --logs ../../logs
```

## Dataset
In order to download the dataset the python package gdown was used, that ignores warnings that occure for large files on Google drive. `pip install gdown`.
The links for a subset of scenes from the non-sequential dataset (folder HD7) and the corresponding filenames can be found in the csv files list_of_files_inHD7.csv. The script that was used to create the csv's can be found [here](https://docs.google.com/spreadsheets/d/1a8Ys_xbKbW9BKdZ-6PHchmeyhDzgHAxYINMd3-h9C2I/edit?usp=sharing). The script can be changed under Tools->Scripteditor. It can only run for 5 minutes and has to be restarted afterwards.
