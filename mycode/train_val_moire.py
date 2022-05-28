# coding = utf8
# /usr/bin/env python

'''
Author: Xjg
Email: 
date: 2021/12/18 下午3:24
desc:
'''
import glob
import os
import shutil
from tqdm import tqdm



def mkdir_img(train_path):
    if not os.path.exists(train_path):
        input_path = os.path.join(train_path, 'input')
        target_path = os.path.join(train_path, 'target')
        os.makedirs(train_path)
        os.makedirs(input_path)
        os.makedirs(target_path)


# path = '/data2/20120017/datasets/testData/testData/source'

test_path = '/data2/20120017/datasets/testData/test'
mkdir_img(test_path)




def train_val_img(val_sum, path, input_name, target_name):
    all_input_img = os.listdir(path)

    print(len(all_input_img))
    choose_img_path = all_input_img[:val_sum]
    print(len(choose_img_path))

    for img_path in tqdm(choose_img_path):

        img_path = os.path.join(path, img_path)

        shutil.move(img_path, os.path.join(test_path, 'input'))
        img_gt_path = img_path.replace(input_name, target_name)
        shutil.move(img_gt_path, os.path.join(test_path, 'target'))




if __name__ == "__main__":
    # train_val_img(50, '/data2/20120017/datasets/moire_train_dataset/images/', 'images', 'gts')
    # train_val_img(10, '/data2/20120017/datasets/ValidationMoire', 'ValidationMoire', 'ValidationClear')
    train_val_img(100, '/data2/20120017/datasets/testData/train/input', 'input', 'target')






