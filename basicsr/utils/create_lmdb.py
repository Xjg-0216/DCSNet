
import argparse
from os import path as osp

from basicsr.utils import scandir
from basicsr.utils.lmdb_util import make_lmdb_from_imgs

def prepare_keys(folder_path, suffix='png'):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(
        list(scandir(folder_path, suffix=suffix, recursive=False)))
    keys = [img_path.split('.{}'.format(suffix))[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys



def create_lmdb_for_csd():
    folder_path = '/data/20120017/20120017/DCSNet-main/datasets/CSD/train/input'
    lmdb_path = '/data/20120017/20120017/DCSNet-main/datasets/CSD/train/input.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'tif')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = '/data/20120017/20120017/DCSNet-main/datasets/CSD/train/target'
    lmdb_path = '/data/20120017/20120017/DCSNet-main/datasets/CSD/train/target.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'tif')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def create_lmdb_for_srrs():
    folder_path = '/data/20120017/20120017/DCSNet-main/datasets/SRRS/train/input'
    lmdb_path = '/data/20120017/20120017/DCSNet-main/datasets/SRRS/train/input.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'tif')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = '/data/20120017/20120017/DCSNet-main/datasets/SRRS/train/target'
    lmdb_path = '/data/20120017/20120017/DCSNet-main/datasets/SRRS/train/target.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'jpg')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def create_lmdb_for_snow100k():
    folder_path = '/data/20120017/20120017/DCSNet-main/datasets/Snow100K/train/input'
    lmdb_path = '/data/20120017/20120017/DCSNet-main/datasets/Snow100K/train/input.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'jpg')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = '/data/20120017/20120017/DCSNet-main/datasets/Snow100K/train/target'
    lmdb_path = '/data/20120017/20120017/DCSNet-main/datasets/Snow100K/train/target.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'jpg')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
