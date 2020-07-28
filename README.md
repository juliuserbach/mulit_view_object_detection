# mulit_view_object_detection

## Project Description
Within this project a network architecture was developed to fuse features from multiple images with different viewpoints on the same scene for the purpose of instance segmentation. In order to combine the information from different images, the camera poses are leveraged using projective geometry. By implementing unprojection and projection layers into the network structure, which can transform the information into a 3D coordinate system, the network is not required to learn the rules of projective geometry.
The performance of the developed network is evaluated on the novel InteriorNet dataset. Implementing projective geometry into the network proves to be an applicable approach and the developed modules build a solid base for future work on this topic.

## Structure

'''
.
├── data
│   ├── InteriorNet
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
'''
