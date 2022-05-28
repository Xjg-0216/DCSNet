# coding = utf8
# /usr/bin/env python

'''
Author: Xjg
Email: 
date: 2021/12/14 下午3:00
desc:
'''


from glob import glob
import cv2
import os
from tqdm import tqdm
import shutil

root_path = '/data/20120017/20120017/Derain3Net-main/datasets/SRRS/train/input/'
save_path_1 = '/data/20120017/20120017/Derain3Net-main/datasets/All/train/input/'
save_path_2 = '/data/20120017/20120017/Derain3Net-main/datasets/All/train/target/'


img_path = glob(os.path.join(root_path, '*.tif'))
print(len(img_path))
# if not os.path.exists(save_path_1):
#     os.makedirs(save_path_1)
# if not os.path.exists(save_path_2):
#     os.makedirs(save_path_2)

i =  58001
for path in tqdm(img_path):
    img_input = cv2.imread(path)
    # img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    cv2.imwrite((save_path_1+'{}.jpg'.format(i)), img_input)
    img_target = cv2.imread(path.replace('input', 'target'))
    # img_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB)
    cv2.imwrite((save_path_2+'{}.jpg'.format(i)), img_target)
    i += 1