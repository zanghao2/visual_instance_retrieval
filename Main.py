# -*- coding: utf-8 -*-
from image_retrieval.Image_Retrieval import *
from Make_Image_FeatureDB import FeatureDB
import os

import pdb


def Make_FeatureDB():

    my_Feature_Sets = FeatureDB()

    # create a video feature sets if there is none
    class_name = '0'

    source_image_dir = '/home/tfl/workspace/dataSet/Visual_Instance_Retrieval/missfresh_goods'

    if os.path.exists(source_image_dir):
        default_image_folder = os.path.join(source_image_dir, class_name)
        my_Feature_Sets.create(class_name, default_image_folder)

    # add
    class_name = '1'
    default_image_folder = os.path.join(source_image_dir, class_name)
    my_Feature_Sets.add(class_name, default_image_folder)

    class_name = '2'
    default_image_folder = os.path.join(source_image_dir, class_name)
    my_Feature_Sets.add(class_name, default_image_folder)

    class_name = '3'
    default_image_folder = os.path.join(source_image_dir, class_name)
    my_Feature_Sets.add(class_name, default_image_folder)

    class_name = '4'
    default_image_folder = os.path.join(source_image_dir, class_name)
    my_Feature_Sets.add(class_name, default_image_folder)

    return 0


def search_image_by_image():
    """
    以图搜图

    :return:
    """
    # TODO: (1)将特征提取写成通用接口；(2)考虑在处理输入图片的时候不改变图片的 宽高比；(3)尝试多层次的 r-mac feature

    # Query Image Info
    query_img_name = 'goods_0233_2.jpg'
    query_img_dir = '/home/tfl/workspace/dataSet/Visual_Instance_Retrieval/missfresh_goods/query_images/'
    query_img_path = os.path.join(query_img_dir, query_img_name)

    content_base_image_retrieval(query_img_path,should_plot_result=True)

if __name__ == "__main__":

    #Make_FeatureDB()

    search_image_by_image()

    pass