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

import mrcnn.model_h as modellib

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
        dataset_dir = "{}/{}".format(dataset_dir, subset)
        self.dataset_dir = dataset_dir
        
        # with open(os.path.join(dataset_dir,'instance_mapping.json')) as json_file:
        #     self.instance_map = json.load(json_file)
        # with open(os.path.join(dataset_dir,'view_mapping.json')) as json_file:
        #     self.view_map = json.load(json_file)
        with open(os.path.join(dataset_dir, 'view_mapping_seq.json')) as json_file:
            self.view_map_seq = json.load(json_file)
        
        #list of class names

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
            else:
                coco = COCO("{}/cocolabel.json".format(image_dir))
                add_path = ''
                self.label_path = 'label0/data'

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
    
#     def load_view(self, n, instance=None, rnd_state=None):
#         """ takes number of views n and outputs n image ids of a random instance
#         fix rnd_state for evaluation purposes
#         """
#         if n < 0:  
#             max_views = 4
#         else:
#             max_views = n
#         LocalProcRandGen = np.random.RandomState(rnd_state)
#         if not instance:
#             instance = LocalProcRandGen.choice(list(self.instance_map.keys()),1)[0]
#             # assure that there are at least n different views of the same instance
#             while np.asarray(self.instance_map[instance]).shape[0] < max_views:
#                 instance = LocalProcRandGen.choice(list(self.instance_map.keys()),1)[0]
#         obj_inst = np.asarray(self.instance_map[instance])
#         num_available_views = obj_inst.shape[0]
#         if num_available_views < max_views:
#             return None
#         views = LocalProcRandGen.choice(range(obj_inst.shape[0]), max_views, replace=False)
#         image_ids = obj_inst[views][:,1]
#         image_ids = image_ids[:n]
#         out = []
#         for image_id in image_ids:
#             image_id = self.image_from_source_map['interior.'+image_id]
#             out.append(image_id)
#         return out

    def load_view(self, n, main_image, rnd_state=None):
        max_views = 5
        LocalProcRandGen = np.random.RandomState(rnd_state)
        secondary_views = np.asarray(self.view_map_seq[main_image])
        views = LocalProcRandGen.choice(range(secondary_views.shape[0]), max_views - 1, replace=False)
        image_ids = secondary_views[views]
        image_ids = image_ids[:n - 1]
        out = [self.image_from_source_map['interior.' + main_image]]
        for image_id in image_ids:
            image_id = self.image_from_source_map['interior.' + image_id]
            out.append(image_id)
        return out

    
    # def load_view(self, n, main_image=None, rnd_state=None):
    #     """ takes number of views n and outputs n image ids of a random instance
    #     fix rnd_state for evaluation purposes
    #     """
    #     if n < 0:
    #         max_views = 4
    #     else:
    #         max_views = n
    #     max_views = 2
    #     LocalProcRandGen = np.random.RandomState(rnd_state)
    #     if not main_image:
    #         main_image = LocalProcRandGen.choice(list(self.view_map.keys()),1)[0]
    #         print(main_image)
    #         # assure that there are at least n different views of the same instance
    #         while np.asarray(self.view_map[main_image]).shape[0] < max_views:
    #             view = LocalProcRandGen.choice(list(self.view_map.keys()),1)[0]
    #     secondary_views = np.asarray(self.view_map[main_image])
    #     num_available_views = secondary_views.shape[0]
    #     if num_available_views < max_views:
    #         return None
    #     views = LocalProcRandGen.choice(range(secondary_views.shape[0]), max_views-1, replace=False)
    #     image_ids = secondary_views[views]
    #     image_ids = image_ids[:n-1]
    #     out = [self.image_from_source_map['interior.'+main_image]]
    #     for image_id in image_ids:
    #         image_id = self.image_from_source_map['interior.'+image_id]
    #         out.append(image_id)
    #     return out

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
                TOP_DOWN_PYRAMID_SIZE = 128
#                 FPN_CLASSIF_FC_LAYERS_SIZE = 128
#                 POST_NMS_ROIS_INFERENCE = 1000
                POST_NMS_ROIS_TRAINING = 500
                PRE_NMS_LIMIT = 1500
                GPU_COUNT = 1
                IMAGES_PER_GPU = 1
                STEPS_PER_EPOCH = 100
                VALIDATION_STEPS = 20
                NUM_CLASSES = len(selected_classes)  # background + num classes
                nvox = 40
                nvox_z = 40
                min_z = 0.5
                max_z = 5.
                GRID_DIST = 5.
                vmin = -2.5
                vmax = 2.5
                vmax_z = 10.
                vmin_z = 1.
                vsize = float(vmax - vmin) / nvox
                vsize_z = float(vmax_z - vmin_z) / nvox_z
                vox_bs = 1
                im_bs = 1
                samples = 25
                NUM_VIEWS = 2 
                RECURRENT = False
                USE_RPN_ROIS = True
                LEARNING_RATE = 0.001
                GRID_REAS = 'ident'
                BACKBONE = 'resnet50'
                VANILLA = False
                WEIGHT_DECAY = 0.0001
                TRANSFORMER = False
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
            min_z = 0.5
            max_z = 5.
            GRID_DIST = 5.
            vmin = -5.
            vmax = 5.
            vsize = float(vmax - vmin) / nvox
            vox_bs = 1
            im_bs = 1
            samples = 25
            NUM_VIEWS = 2
            RECURRENT = False
            USE_RPN_ROIS = True
            LEARNING_RATE = 0.01
            GRID_REAS = 'ident'
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

#     # Load weights
    #model_path_bb = os.path.join(args.logs, '../weights', 'mask_rcnn_interiornet_bb.h5')
#     print("Loading weights ", model_path)
#     from keras.engine import saving
#     import h5py
#     # f = h5py.File(os.path.join(model_path), mode='r')
#     folder, weight_name = os.path.split(model_path)
#     epoch = weight_name[-7:-3]
#     model_path_bb = folder+'/backbone_callb_epoch_{}.h5'.format(epoch)
#     f = h5py.File(os.path.join(model_path_bb), mode='r')
#     #f = h5py.File(COCO_MODEL_PATH, mode='r')
#     for layer in model.keras_model.layers:
#         if layer.name == 'backbone':
#             layers = layer.layers
#             print(layer.__class__.__name__)
#             saving.load_weights_from_hdf5_group_by_name(f, layer.layers)
#             break
#     model_path_bb = os.path.join(ROOT_DIR, 'weights', 'mask_rcnn_interiornet_bb.h5', 'backbone_sep.h5')
#     model.load_weights(model_path_bb, by_name=True)
#     train_layers = 'grid+'
#     layer_regex = {
#             # all layers but the backbone
#             "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
#             "grid+": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(grid_reas\_.*)",
#             "grid+-": r"(mrcnn\_.*)|(rpn\_.*)|(grid_reas\_.*)",
#             "grid_only": r"(grid_reas\_.*)",
#             # From a specific Resnet stage and up
#             "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(grid_reas\_.*)",
#             "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(grid_reas\_.*)",
#             "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(grid_reas\_.*)",
#             # All layers
#             "all": ".*",
#         }
#     if train_layers in layer_regex.keys():
#         layers = layer_regex[train_layers]
#     model.set_trainable(layers)
    print(model_path)
    model.load_weights(model_path, by_name=True, exclude=["mrcnn_bbox_fc", "mrcnn_class_logits", "mrcnn_mask", "fpn_c5p5", "fpn_c4p4", "fpn_c3p3", "fpn_c2p2", "fpn_p5", "fpn_p4", "fpn_p3", "fpn_p2", "rpn_model", "mrcnn_mask_conv1", "mrcnn_class_conv1", "mrcnn_mask_bn1", "mrcnn_mask_conv2", "mrcnn_mask_bn2", "mrcnn_mask_conv3", "mrcnn_mask_bn3", "mrcnn_mask_conv4", "mrcnn_mask_bn4", "mrcnn_mask_deconv"])
#    model.load_weights(model_path, by_name=True)
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
        print("Training all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=301,
                    layers='grid+')
#         Training - Stage 2
#         Finetune layers from ResNet stage 4 and up
#         Training - Stage 3
#         Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=501,
                    layers='4+')

        # Training - Stage 4
#         Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE/10,
                    epochs=751,
                    layers='all')
#         for layer in model.keras_model.layers:
#             if layer.name == 'backbone':
#                 layer.save_weights(os.path.join(model.log_dir, 'backbone_sep.h5'))
#                 break

    elif args.command == "evaluate":
        dataset = InteriorDataset()
        dataset.load_Interior(dataset_dir=args.dataset, subset='test', class_ids=selected_class_list,
                                    NYU40_to_sel_map=NYU40_to_sel_map, selected_classes=selected_classes)
        dataset.prepare()
        
        instance_ids = np.copy(list(dataset.instance_map.keys()))
        view_ids = np.copy(list(dataset.view_map.keys()))
        
        def compute_batch_ap(view_ids):
            max_views = 2
            APs = []
            APs_range = []
            for view_index, view_id in enumerate(view_ids):
                #instance_id = instance_ids[instance_index]               
                image_ids = dataset.load_view(max_views, main_image=view_id, rnd_state=0)
                # skip instance if it has to few views (return of load_views=None)
                if not image_ids:
                    continue
                image_ids = image_ids[:config.NUM_VIEWS]
                #image_pair = image_ids.reshape([-1,config.NUM_VIEWS])
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
#                 im[1,:,:,:] = im[1,:,:,:]*0.
                Rcam = np.stack([Rcam])
                Kmat = np.stack([Kmat])
                # Run object detection
                results = model.detect([im], Rcam, Kmat)
                # Compute AP
                r = results[0]
                AP, precisions, recalls, overlaps =\
                    utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                      r['rois'], r['class_ids'], r['scores'], r['masks'])
                
                AP_range = compute_ap_range(gt_bbox, gt_class_id, gt_mask,
                                      r['rois'], r['class_ids'], r['scores'], r['masks'])
                
                APs.append(AP)
                APs_range.append(AP_range)
                print("meanAP: {}".format(np.mean(APs)))
                print("AP_range: {}".format(AP_range))
            return APs

        # Pick a set of random images
        APs = compute_batch_ap(view_ids[:])
        np.save(model.log_dir, APs)
        print("mAP @ IoU=50: ", np.mean(APs))
        
    elif args.command == "visualize":
        max_views = 2
        dataset = InteriorDataset()
        dataset.load_Interior(dataset_dir=args.dataset, subset='test', class_ids=selected_class_list, 
                                    NYU40_to_sel_map=NYU40_to_sel_map, selected_classes=selected_classes)
        dataset.prepare()
        view_ids = np.copy(list(dataset.view_map.keys()))
        
        num_views_map = {1: 'NV1', 2: 'NV2', 3: 'NV3', 4: 'NV4'}
        SAVE_DIR = os.path.join(ROOT_DIR, 'data/InteriorNet/Results', num_views_map[config.NUM_VIEWS])
        for instance_index, instance_id in enumerate(view_ids):
            #instance_id = instance_ids[instance_index]               
            image_ids = dataset.load_view(max_views, main_image=view_id, rnd_state=1)
            # skip instance if it has to few views (return of load_views=None)
            if not image_ids:
                continue
            # Load image

            print("processing image {} of {}".format(instance_index, instance_ids.size)) 
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

        '''
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