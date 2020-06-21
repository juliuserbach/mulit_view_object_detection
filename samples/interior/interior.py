"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_shapes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import glob
import sys
import math
import random
import numpy as np
import cv2
import imageio
import re
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.callbacks import LearningRateScheduler

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils


# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model50 as modellib

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.log_device_placement = True  # to log device placement (on which device the operation ran)
sess= tf.Session(config=config_tf)
keras.backend.tensorflow_backend.set_session(sess)

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


class InteriorNetConfig(Config):
    """Configuration for training on the InteriorNet dataset.
    Derives from the base Config class and overrides values specific
    to the InterioNet dataset.
    """
    # Give the configuration a recognizable name
    NAME = "InteriorNet"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 40  # background + 40 classes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640


class InteriorDataset(utils.Dataset):
    
    
    def load_Interior(self, dataset_dir, subset, NYU40_to_sel_map, selected_classes, class_ids=None, return_coco=False):
        """Load a subset of the InteriorNet dataset.
        dataset_dir: The root directory of the InteriorNet dataset.
        subset: What to load (train, test, val
        class_ids: If provided, only loads images that have the given classes.
        return_coco: If True, returns the COCO object.
        """
        dataset_dir = "{}/{}".format(dataset_dir, subset)
        self.dataset_dir = dataset_dir
        
        #list of class names

        self.NYU40_to_sel_map = NYU40_to_sel_map
        
        # Save list of class_ids of interest
        self.class_ids = class_ids
        
        # Iterate trough all files in the subset folder
        image_dirs = glob.iglob(os.path.join(dataset_dir,'*'))
        for image_dir in image_dirs:
            if bool(re.search('.json' ,image_dir)):
                continue
            # Select current cocolabe.json file
            coco = COCO("{}/cocolabel.json".format(image_dir))
            head, tail = os.path.split(image_dir)
           
            if not class_ids:
                # All classes
                class_ids = sorted(coco.getCatIds())
                print(class_ids)

            # All images or a subset?
            if class_ids:
                image_ids = []
                for id in class_ids:
                    image_ids.extend(list(coco.getImgIds(catIds=[id])))
                # Remove duplicates
                image_ids = list(set(image_ids))
            else:
                # All images
                image_ids = list(coco.imgs.keys())

            # Add classes
            for i in range(1, len(selected_classes)):
                self.add_class("interior", i, selected_classes[i])

            # Add images
            for i in image_ids:
                id = tail + '_id' + str(i)
                self.add_image(
                    "interior", image_id=id,
                    image_sub_id = i,
                    path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                    width=coco.imgs[i]["width"],
                    height=coco.imgs[i]["height"],
                    annotations=coco.loadAnns(coco.getAnnIds(
                        imgIds=[i], catIds=class_ids, iscrowd=None)),
                    subfolder=tail)
        if return_coco:
            return coco


    def image_reference(self, image_id):
        """Return the path of the image"""
        
        info = self.image_info[image_id]
        if info["source"] == "interior":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
        
    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not an InteriorNet image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "interior":
            return super(self.__class__, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        path = self.image_info[image_id]["path"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        instance_mask_path = os.path.join(self.dataset_dir, self.image_info[image_id]["subfolder"], 
                                   'label0/data',
                                   str(self.image_info[image_id]["image_sub_id"])+'_instance.png')
        nyu_mask_path = os.path.join(self.dataset_dir, self.image_info[image_id]["subfolder"], 
                                   'label0/data',
                                   str(self.image_info[image_id]["image_sub_id"])+'_nyu.png')
        instance_im = imageio.imread(instance_mask_path)
        nyu_im = imageio.imread(nyu_mask_path)
        instance_ids = np.unique(instance_im)
        for instance_id in instance_ids:
            binary_mask = np.where(instance_im == instance_id , True, False)
            class_id = nyu_im[binary_mask]
            # As an error never occured, an unique class id will be assumed
            #class_id = np.unique(class_id)
            #if len(class_id==1):
            if self.NYU40_to_sel_map[class_id[0]] !=0:   
                class_ids.append(self.NYU40_to_sel_map[class_id[0]])
                instance_masks.append(binary_mask)
            #else:
            #    print('Multiple different classes inside one instance mask.')

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    
        

############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on InteriorNet.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on InteriorNet")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/intriornet/",
                        help='Directory of the InterioNet dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--logs', required=False,
                        default='../../logs',
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    
    NYU40_to_sel_map = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 0, 9: 0, 10: 8, 11: 6, 12: 0, 13: 7, 14: 5, 15: 8, 16: 0, 17: 9, 18: 10, 19: 11, 20: 0, 21: 12, 22: 0, 23: 13, 24: 14, 25: 15, 26: 16, 27: 17, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 18, 34: 19, 35: 20, 36: 21, 37: 22, 38: 0, 39: 0, 40: 0}
    selected_classes = ['BG', 'cabinet', 'bed', 'chair', 'sofa', 'table', 
                    'picture', 'blinds', 'shelves', 'dresser', 'pillow', 
                    'mirror',  'clothes','books', 'refrigerator','television', 'paper', 'towel',
                    'toilet', 'sink', 'lamp', 'bathtub', 'bag']
    selected_class_list = [0, 3, 4, 5, 6, 7, 11, 13, 15, 17, 18, 19, 21, 23, 25, 26, 27, 33, 34, 35, 36, 37, 14, 10, 24]

    # Configurations
    if args.command == "train":
        class TrainConfig(InteriorNetConfig):
            TOP_DOWN_PYRAMID_SIZE = 64
#             FPN_CLASSIF_FC_LAYERS_SIZE = 256
            BACKBONE = 'resnet50'    
            STEPS_PER_EPOCH = 5000
            VALIDATION_STEPS = 800 
#             POST_NMS_ROIS_INFERENCE = 2000
#             POST_NMS_ROIS_TRAINING = 1000
#             PRE_NMS_LIMIT = 1500
            NUM_CLASSES = len(selected_classes)  # background + num classes
            GPU_COUNT = 1
            IMAGES_PER_GPU = 2
            LEARNING_RATE = 0.001
        config = TrainConfig()
    else:
        class InferenceConfig(InteriorNetConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
            TOP_DOWN_PYRAMID_SIZE = 64
#             FPN_CLASSIF_FC_LAYERS_SIZE = 128
            BACKBONE = 'resnet50'    
            STEPS_PER_EPOCH = 5000
            VALIDATION_STEPS = 800 
#             POST_NMS_ROIS_INFERENCE = 1000
#             POST_NMS_ROIS_TRAINING = 500
            BACKBONE = 'resnet50'       
            NUM_CLASSES = len(selected_classes)  # background + num classes
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    #print("Loading weights ", model_path)
#     model.load_weights(model_path, by_name=True, exclude=["rpn_model",
#             "mrcnn_class_logits", "mrcnn_bbox_fc",
#             "mrcnn_bbox", "mrcnn_mask", "mrcnn_mask_conv1", "mrcnn_class_conv1"])
    model.load_weights(model_path, by_name=True)
#     model.load_weights(model_path, by_name=True, exclude=["rpn_model", "mrcnn_mask_conv1", "mrcnn_class_conv1", "mrcnn_mask_bn1",  "mrcnn_mask_conv2", "mrcnn_mask_bn2", "mrcnn_class_bn1", "mrcnn_mask_conv3", "mrcnn_mask_bn3", "mrcnn_class_conv2", "mrcnn_class_bn2", "mrcnn_mask_conv4", "mrcnn_mask_bn4", "mrcnn_bbox_fc", "mrcnn_mask_deconv", "mrcnn_class_logits", "mrcnn_mask"])
#     model.load_weights(model_path, by_name=True, exclude=["rpn_model", "mrcnn_mask_conv1", "mrcnn_class_conv1", "fpn_c5p5", "fpn_c4p4", "fpn_c3p3", "fpn_c2p2", "fpn_p5", "fpn_p2", "fpn_p3", "fpn_p4", "mrcnn_mask_bn1",  "mrcnn_mask_conv2", "mrcnn_mask_bn2", "mrcnn_class_bn1", "mrcnn_mask_conv3", "mrcnn_mask_bn3", "mrcnn_class_conv2", "mrcnn_class_bn2", "mrcnn_mask_deconv", "mrcnn_mask_conv4", "mrcnn_mask_bn4", "mrcnn_bbox_fc", "mrcnn_class_logits", "mrcnn_mask"])
    
    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = InteriorDataset()
        dataset_train.load_Interior(dataset_dir=args.dataset, subset='train', class_ids=selected_class_list, 
                                    NYU40_to_sel_map=NYU40_to_sel_map, selected_classes=selected_classes)
        dataset_train.prepare()
        
        dataset_val = InteriorDataset()
        dataset_val.load_Interior(dataset_dir=args.dataset, subset='val', class_ids=selected_class_list, 
                                    NYU40_to_sel_map=NYU40_to_sel_map, selected_classes=selected_classes)
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        #augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***
        def step_decay(epoch):
            initial_lrate = 0.002
            drop = 0.5
            epochs_drop = 3
            lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
            if lrate < 0.001:
                return 0.001
            return lrate
        lrate = LearningRateScheduler(step_decay)
#         tensorboard_callback = tf.keras.callbacks.TensorBoard(model.log_dir, update_freq = 1000)
        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=23,
                    layers='heads',
                    custom_callbacks = [lrate])
        print("Training resnet 4+")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=23,
                    layers='4+')
        print("Training all")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE/10,
                    epochs=35,
                    layers='all')

    elif args.command == "evaluate":
        dataset_val = InteriorDataset()
        dataset_val.load_Interior(dataset_dir=args.dataset, subset='val', class_ids=selected_class_list, 
                                    NYU40_to_sel_map=NYU40_to_sel_map, selected_classes=selected_classes)
        dataset_val.prepare()
        def compute_batch_ap(image_ids):
            APs = []
            for im_num, image_id in enumerate(image_ids):
                # Load image
                print("processing image {} of {}".format(im_num, image_ids.size)) 
                image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                    modellib.load_image_gt(dataset_val, config,
                                           image_id, use_mini_mask=False)
                # Run object detection
                results = model.detect([image], verbose=0)
                # Compute AP
                r = results[0]
                AP, precisions, recalls, overlaps =\
                    utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                      r['rois'], r['class_ids'], r['scores'], r['masks'])
                APs.append(AP)
            return APs

        # Pick a set of random images
        APs = compute_batch_ap(dataset_val.image_ids)
        np.save(model.log_dir, APs)
        print("mAP @ IoU=50: ", np.mean(APs))
        
    
    elif args.command == "visualize":
        dataset_val = InteriorDataset()
        dataset_val.load_Interior(dataset_dir=args.dataset, subset='val', class_ids=selected_class_list, 
                                    NYU40_to_sel_map=NYU40_to_sel_map, selected_classes=selected_classes)
        dataset_val.prepare()
        image_ids = dataset_val.image_ids
        
        obj_inst = np.asarray(self.instance_map[instance]) 
        
        SAVE_DIR = os.path.join(ROOT_DIR, 'data/InteriorNet/data/Results/NV1')
        
        for image_id in image_ids:
            image = dataset_val.load_image(image_id)
            results = model.detect([image], verbose=0)
            r = results[0]
            visualize.save_image(image_name = image_id,
                                 image = image, boxes = r['rois'], 
                                 masks = r['masks'], class_ids = r['class_ids'],
                                 class_names = selected_classes, 
                                 scores = r['scores'], save_dir = SAVE_DIR)

'''
    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = CocoDataset()
        val_type = "val" if args.year in '2017' else "minival"
        coco = dataset_val.load_coco(args.dataset, val_type, year=args.year, return_coco=True, auto_download=args.download)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))

'''