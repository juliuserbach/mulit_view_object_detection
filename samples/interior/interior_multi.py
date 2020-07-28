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
import matplotlib.pyplot as plt
import csv
import json
import re
import tensorflow as tf
import keras
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import Callback


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils


# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import visualize
from samples.interior import classes

import mrcnn.model_multi as modellib

import tensorflow as tf
import keras
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
        _, hd_folder = os.path.split(dataset_dir)
        print(hd_folder)
        self.hd_folder = hd_folder
        dataset_dir = "{}/{}".format(dataset_dir, subset)
        self.dataset_dir = dataset_dir

        self.NYU40_to_sel_map = NYU40_to_sel_map
        
        # Save list of class_ids of interest
        self.class_ids = class_ids
        
        # Iterate trough all files in the subset folder
        image_dirs = glob.iglob(os.path.join(dataset_dir,'*'))
        for image_dir in image_dirs:
            print(image_dir)
            if bool(re.search('.json' ,image_dir)):
                print(image_dir)
                continue
            # Select current cocolabe.json file
            if hd_folder != 'HD7':
                coco = COCO("{}/cocolabel.json".format(os.path.join(image_dir, 'original_1_1')))
                add_path = 'original_1_1'
                self.label_path = 'original_1_1/label0/data'
                with open(os.path.join(dataset_dir, 'view_mapping_seq.json')) as json_file:
                    self.view_map = json.load(json_file)
            else:
                coco = COCO("{}/cocolabel.json".format(image_dir))
                add_path = ''
                self.label_path = 'label0/data'
                with open(os.path.join(dataset_dir, 'view_mapping.json')) as json_file:
                    self.view_map = json.load(json_file)

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
                
            # add camera matrix K
            f = 600
            x0 = 320
            y0 = 320
            K = np.array([[f, 0, x0],
                         [0, f, y0],
                         [0, 0, 1]])
            self.K = K
            
            # open file with camera poses
            if hd_folder != 'HD7':
                path_to_camera_pose = os.path.join(image_dir, 'velocity_angular_1_1', 'cam0_gt.visim')
                camera_file = open(path_to_camera_pose,"r")   
                csvreader = csv.reader(camera_file, delimiter=',')
                # skip header of the file
                next(csvreader)
                # create dict to match timestamp to pose entry
                time_to_pose = {}
                for row in csvreader:
                    time_to_pose.update({int(row[0]): row[1:]})
            else:
                path_to_camera_pose = os.path.join(image_dir, 'cam0.render')
                camera_file = open(path_to_camera_pose,"r")
                csvreader = csv.reader(camera_file, delimiter=' ')
                # skip header lines 
                next(csvreader)
                next(csvreader)
                next(csvreader)
                time_to_pose = {}
                for row in csvreader:
                    time_to_pose.update({int(row[0]): row[1:]})
                
            # Add images
            for i in image_ids:
                image_name = os.path.split(coco.imgs[i]['file_name'])[1][0:-4]
                id = tail + '_id' + image_name
                print(id)
                timestamp = coco.imgs[i]['timestamp'] 
                if hd_folder != 'HD7':
                    x, y, z, qw, qx, qy, qz = [ float(x) for x in time_to_pose[timestamp]]
                    R=np.concatenate((utils.quat2rot([qw, qx, qy, qz]), np.array([[x], [y], [z]])), axis=1)
                else:
                    vec = [ float(x) for x in time_to_pose[timestamp]]
                    R=np.concatenate((utils.vec2rot(np.array(vec)), np.array(vec[1:4]).reshape(3,1)), axis=1)
                self.add_image(
                    "interior", image_id=id,
                    image_sub_id = image_name,
                    path=os.path.join(image_dir, add_path, coco.imgs[i]['file_name']),
                    width=coco.imgs[i]["width"],
                    height=coco.imgs[i]["height"],
                    annotations=coco.loadAnns(coco.getAnnIds(
                        imgIds=i, catIds=class_ids, iscrowd=None)),
                    subfolder=tail,
                    R=R)
                
        
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
                                   self.label_path,
                                   str(self.image_info[image_id]["image_sub_id"])+'_instance.png')
        nyu_mask_path = os.path.join(self.dataset_dir, self.image_info[image_id]["subfolder"], 
                                   self.label_path,
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


    def load_depth(self, image_id, config):
        """returns the depth image of the correpsonding image_id"""
        depth_path = os.path.join(self.dataset_dir, self.image_info[image_id]["subfolder"], 'depth0/data',
                     str(self.image_info[image_id]["image_sub_id"])+'.png')
        depth_image = imageio.imread(depth_path)
        depth_image = depth_image[:, :, np.newaxis]
        depth_image, _, _, _, _ = utils.resize_image(
            depth_image,
            min_dim=20,
            min_scale=config.IMAGE_MIN_SCALE,
            max_dim=20,
            mode=config.IMAGE_RESIZE_MODE)
        depth_image = depth_image[:,:,0]
        return depth_image


    def load_R(self, image_id):
        """returns the pose as R|t with shape 3x4
        """
        image_info = self.image_info[image_id]
        return image_info["R"]


    def load_view(self, n, main_image, rnd_state=None):
        max_views = 5
        if self.hd_folder != 'HD7':
            num_skip = 5
            LocalProcRandGen = np.random.RandomState(rnd_state)
            secondary_views = np.asarray(self.view_map[main_image])
            #views = LocalProcRandGen.choice(range(secondary_views.shape[0]), max_views - 1, replace=False)
            #image_ids = secondary_views[views]
            image_ids = secondary_views[::-1]
            image_ids = image_ids[num_skip:n*num_skip:num_skip]
            out = [self.image_from_source_map['interior.' + main_image]]
            for image_id in image_ids:
                image_id = self.image_from_source_map['interior.' + image_id]
                out.append(image_id)
            return out
        else:
            LocalProcRandGen = np.random.RandomState(rnd_state)
            if not main_image:
                main_image = LocalProcRandGen.choice(list(self.view_map.keys()), 1)[0]
                print(main_image)
                # assure that there are at least n different views of the same instance
                while np.asarray(self.view_map[main_image]).shape[0] < max_views:
                    view = LocalProcRandGen.choice(list(self.view_map.keys()), 1)[0]
            secondary_views = np.asarray(self.view_map[main_image])
            num_available_views = secondary_views.shape[0]
            if num_available_views < max_views:
                return None
            views = LocalProcRandGen.choice(range(secondary_views.shape[0]), max_views - 1, replace=False)
            image_ids = secondary_views[views]
            image_ids = image_ids[:n - 1]
            out = [self.image_from_source_map['interior.' + main_image]]
            for image_id in image_ids:
                image_id = self.image_from_source_map['interior.' + image_id]
                out.append(image_id)
            return out

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

    NYU40_to_sel_map = classes.NYU40_to_sel_map
    selected_classes = classes.selected_classes
    selected_class_list = classes.selected_class_list
    # Configurations
    if args.command == "train":
        class TrainConfig(InteriorNetConfig):
            TOP_DOWN_PYRAMID_SIZE = 64
            POST_NMS_ROIS_TRAINING = 500
            PRE_NMS_LIMIT = 1500
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            STEPS_PER_EPOCH = 100
            VALIDATION_STEPS = 20
            NUM_CLASSES = len(selected_classes)  # background + num classes
            nvox = 40
            nvox_z = 40
            vmin = -2.5
            vmax = 2.5
            vmax_z = 10.
            vmin_z = 1.
            vsize = float(vmax - vmin) / nvox
            vsize_z = float(vmax_z - vmin_z) / nvox_z
            samples = 20
            NUM_VIEWS = 2
            USE_RPN_ROIS = True
            LEARNING_RATE = 0.001
            GRID_REAS = 'conv3d'
            BACKBONE = 'resnet50'
            VANILLA = False
        config = TrainConfig()
    else:
        class InferenceConfig(InteriorNetConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            TOP_DOWN_PYRAMID_SIZE = 64
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            NUM_CLASSES = len(selected_classes)  # background + num classes
            nvox = 40
            nvox_z = 40
            GRID_DIST = 5.
            vmin = -2.5
            vmax = 2.5
            vmax_z = 10.
            vmin_z = 1.
            vsize = float(vmax - vmin) / nvox
            vsize_z = float(vmax_z - vmin_z) / nvox_z
            samples = 20
            NUM_VIEWS = 1
            USE_RPN_ROIS = True
            LEARNING_RATE = 0.01
            GRID_REAS = 'conv3d'
            BACKBONE = 'resnet50'
            VANILLA = False
            
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

    print(model_path)
#     model.load_weights(model_path, by_name=True, exclude=["mrcnn_bbox_fc", "mrcnn_class_logits", "mrcnn_mask", "fpn_c5p5", "fpn_c4p4", "fpn_c3p3", "fpn_c2p2", "fpn_p5", "fpn_p4", "fpn_p3", "fpn_p2", "rpn_model", "mrcnn_mask_conv1", "mrcnn_class_conv1", "mrcnn_mask_bn1", "mrcnn_mask_conv2", "mrcnn_mask_bn2", "mrcnn_mask_conv3", "mrcnn_mask_bn3", "mrcnn_mask_conv4", "mrcnn_mask_bn4", "mrcnn_mask_deconv"])
    model.load_weights(model_path, by_name=True)
#     

    
    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = InteriorDataset()
        dataset_train.load_Interior(dataset_dir=args.dataset, subset='train', class_ids=selected_class_list, 
                                    NYU40_to_sel_map=NYU40_to_sel_map, selected_classes=selected_classes)
        dataset_train.prepare()
        print(dataset_train.image_from_source_map)
        
        dataset_val = InteriorDataset()
        dataset_val.load_Interior(dataset_dir=args.dataset, subset='val', class_ids=selected_class_list, 
                                    NYU40_to_sel_map=NYU40_to_sel_map, selected_classes=selected_classes)
        dataset_val.prepare()
        def step_decay(epoch):
            initial_lrate = 0.002
            drop = 0.5
            epochs_drop = 1
            lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
            if lrate < 0.001:
                return 0.001
            return lrate
        lrate = LearningRateScheduler(step_decay)
        
        
        # Image Augmentation
        # Right/Left flip 50% of the time
#         augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***
        # Training - Stage 1
        print("Training grid and up layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=301,
                    layers='grid+')

#         Training - Stage 2
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=4001,
                    layers='4+')


        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE/10,
                    epochs=4501,
                    layers='all')


    elif args.command == "evaluate":
        dataset = InteriorDataset()
        dataset.load_Interior(dataset_dir=args.dataset, subset='val', class_ids=selected_class_list,
                                    NYU40_to_sel_map=NYU40_to_sel_map, selected_classes=selected_classes)
        dataset.prepare()
        view_ids = np.copy(list(dataset.view_map.keys()))
        
        def compute_batch_ap(view_ids):
            max_views = 5
            APs = []
            APs_range = []
            for view_index, view_id in enumerate(view_ids):
                image_ids = dataset.load_view(max_views, main_image=view_id, rnd_state=0)
                # skip instance if it has to few views (return of load_views=None)
                if not image_ids:
                    continue
                image_ids = image_ids[:config.NUM_VIEWS]
                # Load image
                print("processing image {} of {}".format(view_index, view_ids.size))
                image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                    modellib.load_image_gt(dataset, config,
                                           image_ids[0], use_mini_mask=False)
                im = []
                Rcam = []
                Kmat = dataset.K
                for image_id in image_ids:
                    image = dataset.load_image(image_id)
                    image, _, _, _, _ = utils.resize_image(
                        image,
                        min_dim=config.IMAGE_MIN_DIM,
                        min_scale=config.IMAGE_MIN_SCALE,
                        max_dim=config.IMAGE_MAX_DIM,
                        mode=config.IMAGE_RESIZE_MODE)
                    im.append(image)
                    Rcam.append(dataset.load_R(image_id))

                im = np.stack(im)
                Rcam = np.stack([Rcam])
                Kmat = np.stack([Kmat])
                # Run object detection
                results = model.detect([im], Rcam, Kmat)
                # Compute AP
                r = results[0]
                AP, precisions, recalls, overlaps =\
                    utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                      r['rois'], r['class_ids'], r['scores'], r['masks'])

                #AP_range = utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
                 #                     r['rois'], r['class_ids'], r['scores'], r['masks'], verbose=0)
                
                APs.append(AP)
                #APs_range.append(AP_range)
                print("meanAP: {}".format(np.mean(APs)))
                #print("AP_range: {}".format(np.mean(APs_range)))
            return APs, APs_range

        # Pick a set of random images
        APs, APs_range = compute_batch_ap(view_ids[:])
        np.save(model.log_dir, APs)
        np.save(model.log_dir, APs_range)
        print("mAP @ IoU=50: ", np.mean(APs))
        print("mAP_range @IoU[0.5;0.95]: {}".format(np.mean(APs_range)))
        
    elif args.command == "visualize":
        max_views = 5
        dataset = InteriorDataset()
        dataset.load_Interior(dataset_dir=args.dataset, subset='val', class_ids=selected_class_list, 
                                    NYU40_to_sel_map=NYU40_to_sel_map, selected_classes=selected_classes)
        dataset.prepare()
        view_ids = np.copy(list(dataset.view_map.keys()))
        
        num_views_map = {1: 'NV1', 2: 'NV2', 3: 'NV3', 4: 'NV4'}
        SAVE_DIR = os.path.join(ROOT_DIR, 'data/InteriorNet/Results', num_views_map[config.NUM_VIEWS])
        for view_index, view_id in enumerate(view_ids):
            image_ids = dataset.load_view(max_views, main_image=view_id, rnd_state=1)
            # skip instance if it has to few views (return of load_views=None)
            if not image_ids:
                continue
            # Load image

            print("processing image {} of {}".format(view_index, view_ids.size)) 
            image = dataset.load_image(image_ids[0])
            im = []
            Rcam = []
            Kmat = dataset.K
            image_ids = image_ids[:config.NUM_VIEWS]
            for image_id in image_ids:
                image = dataset.load_image(image_id)
                im.append(image)
                Rcam.append(dataset.load_R(image_id))

            im = np.stack(im)
            Rcam = np.stack([Rcam])
            Kmat = np.stack([Kmat])
            # Run object detection
            results = model.detect([im], Rcam, Kmat)
            r = results[0]
            visualize.save_image(image_name = image_ids[0], 
                                 image = im[0], boxes = r['rois'],
                                 masks = r['masks'], class_ids = r['class_ids'], 
                                 class_names = selected_classes, 
                                 scores = r['scores'], save_dir = SAVE_DIR)
