# coding = utf8
# /usr/bin/env python

'''
Author: Xjg
Email: 
date: 2021/12/13 下午3:35
desc:
'''

import cv2
from glob import glob

multi_stage_imgs_path = '/data/20120017/20120017/Derain3Net-main/demo/demo_prm/'
multi_stage_img_path = glob(multi_stage_imgs_path+'*.tif')
print('multi_stage:', len(multi_stage_img_path))



stage1 = cv2.imread(multi_stage_imgs_path+'demo_stage1.tif')
# stage1 = cv2.cvtColor(stage1, cv2.COLOR_BGR2RGB)
stage2 = cv2.imread(multi_stage_imgs_path+'demo_stage2.tif')
# stage2 = cv2.cvtColor(stage2, cv2.COLOR_BGR2RGB)
stage3 = cv2.imread(multi_stage_imgs_path+'demo_stage3.tif')
# stage3 = cv2.cvtColor(stage3, cv2.COLOR_BGR2RGB)
orgin = cv2.imread('/data/20120017/20120017/Derain3Net-main/demo/611.tif')
# orgin = cv2.cvtColor(orgin, cv2.COLOR_BGR2RGB)
mask1 = orgin - stage1
cv2.imwrite('./mask1.jpg', mask1)
mask2 = orgin - stage2
cv2.imwrite('./mask2.jpg', mask2)
mask3 = orgin - stage3
cv2.imwrite('./mask3.jpg', mask3)
