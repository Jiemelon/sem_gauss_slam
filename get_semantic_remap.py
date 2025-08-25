import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from torch.utils.data import Dataset

import numpy as np
import torch
from tqdm import tqdm
import cv2

class Replica_generated(Dataset):
    def __init__(self, data_dir, img_h=None, img_w=None, sample_step=1, test_step=1, mode='train'):
        self.rgb_dir = os.path.join(data_dir, "rgb")
        self.semantic_class_dir = os.path.join(data_dir, "semantic_class")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_h = img_h
        self.img_w = img_w

        self.data_dir = data_dir

        self.rgb_list = sorted(glob.glob(self.rgb_dir + '/rgb_*.png'),
                               key=lambda file_name: int(file_name.split("_")[-1][:-4]))
        self.semantic_list = sorted(glob.glob(self.semantic_class_dir + '/semantic_class_*.png'),
                                    key=lambda file_name: int(file_name.split("_")[-1][:-4]))

        self.mode = mode

        step = sample_step
        valid_data_num = len(self.rgb_list)
        self.valid_data_num = valid_data_num
        total_ids = range(valid_data_num)

        train_ids = total_ids[::step]
        self.train_ids = train_ids
        self.train_num = len(train_ids)

        self.train_samples = {'semantic_raw': [],
                              'semantic_remap': []}

        self.semantic_classes = set()
        for file in tqdm(self.semantic_list, desc="Reading images"):
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            unique_labels = np.unique(img)
            self.semantic_classes.update(unique_labels)

        self.semantic_classes = np.array(list(self.semantic_classes)).astype(np.uint8)
        self.num_semantic_class = self.semantic_classes.shape[0]
        print("Number of semantic classes: ", self.num_semantic_class)

        # training samples: 存具体值(全部)
        print("train_num: ", self.train_num)
        for idx in tqdm(train_ids, desc="train data"):
            semantic = cv2.imread(self.semantic_list[idx], cv2.IMREAD_UNCHANGED)
            semantic_remap = semantic.copy()
            for i in range(self.num_semantic_class):
                semantic_remap[semantic == self.semantic_classes[i]] = i
            semantic_data = semantic_remap.astype(np.uint8)
            self.train_samples["semantic_remap"].append(semantic_data)

        os.makedirs(os.path.join(data_dir, 'semantic_remap'), exist_ok=True)
        print(os.path.join(data_dir, 'semantic_remap', str(0)+'.png'))
        for idx in range(self.train_num):
            cv2.imwrite(os.path.join(data_dir, 'semantic_remap', 'semantic_remap_'+str(idx)+'.png'), self.train_samples['semantic_remap'][idx])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="./configs/get_remap.yaml",
                        help='config file name.')

    args = parser.parse_args()
    import yaml
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    scannet_data = Replica_generated(data_dir=config["dataset_dir"],
                                img_h=config["height"],
                                img_w=config["width"],
                                sample_step=config["sample_step"],
                                test_step=config["test_step"],
                                mode='train')
