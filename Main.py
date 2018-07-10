# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')

from image_retrieval.Image_Retrieval import *
from Make_Image_FeatureDB import FeatureDB
from src.Evaluate import *
import os

import pdb


def Make_FeatureDB():

    my_Feature_Sets = FeatureDB()

    # create a video feature sets if there is none
    class_name = '0'

    source_image_dir = '/data/data/instance_search/libimg'

    if os.path.exists(source_image_dir):
        default_image_folder = os.path.join(source_image_dir, class_name)
        my_Feature_Sets.create(class_name, default_image_folder)

    # add
    for m in range(1,25):
        class_name = '%d'%m
        default_image_folder = os.path.join(source_image_dir, class_name)
        print default_image_folder,class_name
        my_Feature_Sets.add(class_name, default_image_folder)

    # class_name = '2'
    # default_image_folder = os.path.join(source_image_dir, class_name)
    # my_Feature_Sets.add(class_name, default_image_folder)

    # class_name = '3'
    # default_image_folder = os.path.join(source_image_dir, class_name)
    # my_Feature_Sets.add(class_name, default_image_folder)

    # class_name = '4'
    # default_image_folder = os.path.join(source_image_dir, class_name)
    # my_Feature_Sets.add(class_name, default_image_folder)

    return 0


def search_image_by_image():
    """
    以图搜图

    :return:
    """
    # TODO: (1)将特征提取写成通用接口(完成)；(2)考虑在处理输入图片的时候不改变图片的 宽高比；(3)尝试多层次的 r-mac feature（初步测试，结果变换不大）

    # Query Image Info
    query_img_dir = '/data/data/instance_search/queryimg/'
    dirlists = os.listdir(query_img_dir)

     # Init Feature Extractor
    mFeature_Extractor = FeatureExtractor()

    # Get Feature Sets
    mFeature_Sets = FeatureDB()
    mFeature_Sets.load(default_feature_database_sets_path)

    topN =5 
    thre = 0.4
    classNum = 26
    evl_show = Evaluation(classNum,topN,thre)
    
    for m in dirlists:
        label = m
        imgdir = os.path.join(query_img_dir,m,'images')
        if not os.path.exists(imgdir):
            continue
        imglists = os.listdir(imgdir)
       
        for n,name in enumerate(imglists):
            query_img_name = name
            query_img_path = os.path.join(imgdir, query_img_name)
            print n,query_img_path 
            # result_class, result_image_path, result_conf = content_base_image_retrieval(query_img_path)
            suffix = os.path.splitext(query_img_path)[1]
            if suffix == '.jpg' or suffix == '.png' or suffix == '.jpeg':

                topN = 8
                result_class, result_image_path, result_conf = search_by_image(mFeature_Extractor, mFeature_Sets, query_img_path, top=topN)
                evl_show.CalEval(result_class,result_conf,label)
        evl_show.GetResult()
    

if __name__ == "__main__":

    #Make_FeatureDB()

    search_image_by_image()

    pass