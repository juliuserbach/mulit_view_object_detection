import os
import glob
import csv
import gdown
from zipfile import ZipFile

ROOT_DIR = os.path.abspath('../../')

list_csv = os.path.join(ROOT_DIR, 'data/InteriorNet', 'list_of_files_inHD7.csv')


subsets = ['train', 'val', 'test']
size_of = {'train': 500, 'val': 60, 'test': 60 }
dataset = 'data/InteriorNet/data/HD7'
downloaded_scenes = []
for subset in subsets:
    downloaded_scenes = downloaded_scenes + os.listdir(os.path.join(ROOT_DIR, dataset, subset))
print(downloaded_scenes)
with open(list_csv, 'r') as csvfile:
    scene_reader = csv.reader(csvfile, delimiter=',')
    for subset in subsets:
        for scene in scene_reader:
            name = scene[0][:-4] # cut out .zip
            url = scene[1]
            print(name+'.zip')
            if name in downloaded_scenes:
                continue
            output = os.path.join(ROOT_DIR, 'data/InteriorNet/data/HD7', subset, name+'.zip')
            print(output)
            gdown.download(url, output, quiet=False) 
            with ZipFile(output, 'r') as zipObj:
               # Extract all the contents of zip file in different directory
               zipObj.extractall(os.path.join(ROOT_DIR, dataset, subset))
            os.remove(output)
            downloaded_files = os.listdir(os.path.join(ROOT_DIR, 'data/InteriorNet/data/HD7', subset))
            if len(downloaded_files) > size_of[subset]:
                break