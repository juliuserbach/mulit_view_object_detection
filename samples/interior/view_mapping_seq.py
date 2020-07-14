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


subsets = ["train", "val"]
DATASET_DIR = "data/InteriorNet/data/HD1"
add_path = 'original_1_1'
label_path = 'original_1_1/label0/data'

NYU40_to_sel_map = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 0, 9: 0, 10: 8, 11: 6, 12: 0, 13: 7, 14: 5, 15: 8, 16: 0, 17: 9, 18: 10, 19: 11, 20: 0, 21: 12, 22: 0, 23: 13, 24: 14, 25: 15, 26: 16, 27: 17, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 18, 34: 19, 35: 20, 36: 21, 37: 22, 38: 0, 39: 0, 40: 0}
class_ids = [0, 3, 4, 5, 6, 7, 11, 13, 15, 17, 18, 19, 21, 23, 25, 26, 27, 33, 34, 35, 36, 37, 14, 10, 24]
for subset in subsets:
    # make one mapping for all scenes in one subset(train, val, test) keys are like "scene_name_id"timestep""
    mapping = {}
    scene_paths = glob.glob(os.path.join(ROOT_DIR, DATASET_DIR, subset,'*'))
    for scene_path in scene_paths:
        if bool(re.search('.json' , scene_path)):
            continue

        scene_name = os.path.basename(scene_path)
        coco = COCO("{}/cocolabel.json".format(os.path.join(scene_path, 'original_1_1')))
        #image_ids = list(coco.imgs.keys())
        image_ids = []
        for id in class_ids:
            image_ids.extend(list(coco.getImgIds(catIds=[id])))
        # Remove duplicates
        image_ids = list(set(image_ids))
        view_range = 20
        for i, image_id in enumerate(image_ids[view_range:]):
            if image_ids[i+view_range] - image_ids[i] > view_range + 10:
                continue
            timestamp = lambda x: "{:019d}".format(coco.imgs[x]['timestamp'])
            # instance_mask_path = os.path.join(scene_path,
            #                                   label_path,
            #                                   timestamp(image_id) + '_instance.png')
            # nyu_mask_path = os.path.join(scene_path,
            #                              label_path,
            #                              timestamp(image_id) + '_nyu.png')
            # # Load images
            # instance_im = imageio.imread(instance_mask_path)
            # nyu_im = imageio.imread(nyu_mask_path)
            #
            # # check if objects of interest are present in the image
            # instance_ids = np.unique(instance_im)
            # valid_image = False
            # for instance_id in instance_ids:
            #     binary_mask = np.where(instance_im == instance_id, True, False)
            #     class_id = nyu_im[binary_mask]
            #     if NYU40_to_sel_map[class_id[0]] != 0:
            #         valid_image = True
            #         break
            # if valid_image:
            list_of_neighbors = [scene_name + '_id' + timestamp(j) for j in image_ids[i:i+view_range] if j != image_id]
            print(image_id, len(list_of_neighbors))
            mapping.update({scene_name + '_id' + timestamp(image_id): list_of_neighbors})

    with open('{}.json'.format(os.path.join(ROOT_DIR, DATASET_DIR, subset, 'view_mapping_seq')), 'w') as outfile:
        json.dump(mapping, outfile)



