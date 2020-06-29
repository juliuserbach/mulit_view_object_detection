import os
import glob
import imageio
import csv
import numpy as np
import json
import re


ROOT_DIR = os.path.abspath("../../")

subsets = ["train", "val", "test"]
DATASET_DIR = "data/InteriorNet/data/HD7"
_, hd_folder = os.path.split(DATASET_DIR)
print(hd_folder)

if hd_folder != 'HD7':
    add_path = 'original_1_1'
    label_path = 'original_1_1/label0/data'
else:
    add_path = ''
    label_path = 'label0/data'
    
NYU40_to_sel_map = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 0, 9: 0, 10: 8, 11: 6, 12: 0, 13: 7, 14: 5, 15: 8, 16: 0, 17: 9, 18: 10, 19: 11, 20: 0, 21: 12, 22: 0, 23: 13, 24: 14, 25: 15, 26: 16, 27: 17, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 18, 34: 19, 35: 20, 36: 21, 37: 22, 38: 0, 39: 0, 40: 0}

for subset in subsets:
    # make one mapping for all scenes in one subset(train, val, test) keys are like "scene_name_instance_id"
    mapping = {}
    view_count = 0
    scene_paths = glob.glob(os.path.join(ROOT_DIR, DATASET_DIR, subset,'*'))
    
    header = ["instance_id", "class_id", "frame_id"]
    for scene_path in scene_paths:
        if bool(re.search('.json' ,scene_path)):
            continue
        image_names = os.listdir(os.path.join(scene_path, add_path, 'cam0/data'))
        for image_name in image_names:
            # read timestamp as string
            timestamp = image_name[:-4]
            # Get paths to images
            instance_mask_path = os.path.join(scene_path, 
                                              label_path,
                                              timestamp+'_instance.png')
            nyu_mask_path = os.path.join(scene_path, 
                                         label_path,
                                         timestamp+'_nyu.png')
            # Load images
            instance_im = imageio.imread(instance_mask_path)
            nyu_im = imageio.imread(nyu_mask_path)

            # Get scene id from path
            _, scene_name = os.path.split(scene_path)


            instance_ids = np.unique(instance_im)
            for instance_id in instance_ids:
                binary_mask = np.where(instance_im == instance_id , True, False)
                class_id = nyu_im[binary_mask]

                if NYU40_to_sel_map[class_id[0]] !=0:   
                    view_count += 1
                    if scene_name+'_'+str(instance_id) in mapping:
                        mapping[scene_name+'_'+str(instance_id)].append([NYU40_to_sel_map[class_id[0]], scene_name+'_id'+timestamp])
                    else:
                        mapping.update({scene_name+'_'+str(instance_id): [[NYU40_to_sel_map[class_id[0]], scene_name+'_id'+timestamp]]}) ###

    print("The {}-set has {} views.".format(subset, view_count))                        
    with open('{}.json'.format(os.path.join(ROOT_DIR, DATASET_DIR, subset, 'instance_mapping')), 'w') as outfile:
        json.dump(mapping, outfile)

                            
        
