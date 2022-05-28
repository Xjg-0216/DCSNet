# coding = utf8
# /usr/bin/env python

'''
Author: Xjg
Email: 
date: 2021/12/15 下午3:19
desc:
'''
import cv2
import os
from glob import glob
from tqdm import tqdm
import shutil

path = '/data2/20120017/datasets/testData/train/target/'

#
# if not os.path.exists(save_path):
#     os.makedirs(save_path)



imgs_path = glob(os.path.join(path, '*_target*'))
print(len(imgs_path))
for img_path in tqdm(imgs_path):
    # img_name = img_path.split('/')[-1].replace('jpg', 'png')
    # img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(os.path.join(path, img_name), img)
    # shutil.move(img_path, save_path)
    new_path = img_path[:-11]+'.png'
    os.rename(img_path, new_path)