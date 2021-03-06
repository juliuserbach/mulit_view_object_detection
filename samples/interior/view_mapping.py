import os
import glob
import imageio
import csv
import numpy as np
import json
import re
import sys
from pycocotools.coco import COCO

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils
from samples.interior import classes


ROOT_DIR = os.path.abspath("../../")

subsets = ["train", "val", "test"]
DATASET_DIR = "data/InteriorNet/data/HD1"
_, hd_folder = os.path.split(DATASET_DIR)
print(hd_folder)

if hd_folder != 'HD7':
    add_path = 'original_1_1'
    label_path = 'original_1_1/label0/data'
else:
    add_path = ''
    label_path = 'label0/data'
    
NYU40_to_sel_map = classes.NYU40_to_sel_map

    
    
for subset in subsets:
    # make one mapping for all scenes in one subset(train, val, test) keys are like "scene_name_id"timestep""
    mapping = {}
    scene_paths = glob.glob(os.path.join(ROOT_DIR, DATASET_DIR, subset,'*'))
    for scene_path in scene_paths:
        if bool(re.search('.json' ,scene_path)):
            continue
        scene_name = os.path.basename(scene_path)

        # get coco object from annotation file
        if hd_folder != 'HD7':
            coco = COCO("{}/cocolabel.json".format(os.path.join(scene_path, 'original_1_1')))
        else:
            coco = COCO("{}/cocolabel.json".format(scene_path))


        # get all image ids from coco object
        image_ids = list(coco.imgs.keys())
        image_names = os.listdir(os.path.join(scene_path, add_path, 'cam0/data'))
        # map timestamp to pose R
        if hd_folder != 'HD7':
            path_to_camera_pose = os.path.join(scene_path, 'velocity_angular_1_1/cam0.render')
        else:
            path_to_camera_pose = os.path.join(scene_path, 'cam0.render')
        camera_file = open(path_to_camera_pose,"r")   
        csvreader = csv.reader(camera_file, delimiter=' ')
        # skip header lines 
        next(csvreader)
        next(csvreader)
        next(csvreader)
        time_to_pose = {}
        for row in csvreader:
            time_to_pose.update({int(row[0]): row[1:]})

        # camera matrix K
        f = 600
        x0 = 320
        y0 = 320
        K = np.array([[f, 0, x0],
                     [0, f, y0],
                     [0, 0, 1]])

        vmax = 5.
        vmin = -5.
        nvox = 10
        vsize = float(vmax - vmin) / nvox
        grid_range = np.arange(vmin + vsize / 2.0, vmax,
                                      vsize)
        grid_dist = 600/320 * vmax
        grid_dist = 6.
        
        #select images with objects of interest inside
        valid_image = False     
        
        for i in image_ids:
            #timestamp = image_name[:-4]
            timestamp = coco.imgs[i]['timestamp']
            if hd_folder != 'HD7':
                timestamp_path = "{:019d}".format(timestamp)
            else:
                timestamp_path = timestamp
            instance_mask_path = os.path.join(scene_path,
                                                  label_path,
                                                  str(timestamp_path)+'_instance.png')
            nyu_mask_path = os.path.join(scene_path, 
                                         label_path,
                                         str(timestamp_path)+'_nyu.png')
            # Load images
            instance_im = imageio.imread(instance_mask_path)
            nyu_im = imageio.imread(nyu_mask_path)

            # check if objects of interest are present in the image
            instance_ids = np.unique(instance_im)
            instance_ids_of_interest = []
            for instance_id in instance_ids:
                binary_mask = np.where(instance_im == instance_id , True, False)
                class_id = nyu_im[binary_mask]
                if NYU40_to_sel_map[class_id[0]] !=0: 
                    valid_image = True
                    instance_ids_of_interest.append(instance_id)
                    
            # get R|t
            vec = [ float(x) for x in time_to_pose[timestamp]]
            R=np.concatenate((utils.vec2rot(np.array(vec)), np.array(vec[1:4]).reshape(3,1)), axis=1)
            grid_position = np.dot(R, np.array([0., 0., grid_dist, 1.]).reshape([4,1]))
            grid = np.stack(
                np.meshgrid(grid_range + grid_position[0],
                            grid_range + grid_position[1],
                            grid_range + grid_position[2]))
            rs_grid = np.reshape(grid, [3, -1])
            nV = rs_grid.shape[1]
            rs_grid = np.concatenate([rs_grid, np.ones([1, nV])], axis=0)

            for j in image_ids:
                if j==i or not valid_image:
                    continue
                timestamp = coco.imgs[j]['timestamp']
                if hd_folder != 'HD7':
                    timestamp_path = "{:019d}".format(timestamp)
                else:
                    timestamp_path = timestamp
                instance_mask_path = os.path.join(scene_path, 
                                                  label_path,
                                                  str(timestamp_path)+'_instance.png')
                nyu_mask_path = os.path.join(scene_path, 
                                             label_path,
                                             str(timestamp_path)+'_nyu.png')
                # Load images
                instance_im = imageio.imread(instance_mask_path)
                nyu_im = imageio.imread(nyu_mask_path)

                # check if objects of interest are present in the image
                instance_ids_j = np.unique(instance_im)
                instance_ids_of_interest_j = []
                for instance_id in instance_ids_j:
                    binary_mask = np.where(instance_im == instance_id , True, False)
                    class_id = nyu_im[binary_mask]
                    if NYU40_to_sel_map[class_id[0]] !=0: 
                        instance_ids_of_interest_j.append(instance_id)
                
                # check if the views see the same instance of interest
                valid_secondary = False
                for instance_id in instance_ids_of_interest:
                    if instance_id in instance_ids_of_interest_j:
                        valid_secondary = True
                        break
                
                if valid_secondary:                
                    # get R|t
                    vec = [ float(x) for x in time_to_pose[timestamp]]
                    R=np.concatenate((utils.vec2rot(np.array(vec)), np.array(vec[1:4]).reshape(3,1)), axis=1)
                    Rt = np.transpose(R[:, :3])
                    tr = R[ :, 3].reshape((3,1))
                    R = np.concatenate([Rt, -np.matmul(Rt, tr)], axis=1)
                    KRcam = np.matmul(K, R)
                    # Project grid/
                    im_p = np.matmul(np.reshape(KRcam, [-1, 4]), rs_grid) 
                    im_x, im_y, im_z = im_p[::3, :], im_p[1::3, :], im_p[2::3, :]
                    im_x = (im_x / im_z) 
                    im_y = (im_y / im_z)
                    x_inside = np.where(np.logical_and(im_x > 0, im_x < 640), True, False)
                    y_inside = np.where(np.logical_and(im_y > 0, im_y < 480), True, False)
                    voxel_visible = np.logical_and(y_inside, x_inside)
                    num_visible = voxel_visible.sum()
                    if num_visible/nvox**3 > 0.2:
                        print("percentage of overlap: {}".format(num_visible/nvox**3))
                        if scene_name+'_id'+str(i) in mapping:
                            mapping[scene_name+'_id'+str(i)].append(scene_name+'_id'+str(j))
                        else:
                            mapping.update({scene_name+'_id'+str(i): [scene_name+'_id'+str(j)]}) 
    secondary_views = list(mapping.values())
    print(secondary_views)
    view_count = 0
    for secondary_view in secondary_views:
        if len(secondary_view) >= 4:
            view_count += 1
    print("The {}-set has {} views.".format(subset, view_count))       
    with open('{}.json'.format(os.path.join(ROOT_DIR, DATASET_DIR, subset, 'view_mapping')), 'w') as outfile:
        json.dump(mapping, outfile)
        