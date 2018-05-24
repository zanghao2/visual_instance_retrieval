# -*- coding: utf-8 -*-
import glob
import os

import cv2
import numpy as np

os.environ['GLOG_minloglevel'] = '2'
import caffe
from src.image_feature_extractor import Feature_Extractor
from Make_Image_FeatureDB import Feature_DB
from util import sec_to_hms
from src.image_scoring import Image_Scoring

import pdb

from conf import *

# set gpu mode
caffe.set_mode_gpu()


def content_base_image_retrieval(query, should_plot_result=False):
    """
    Main function of VIR

    :param query: Query image path
    :return:
    """

    # Init Feature Extractor
    mFeature_Extractor = Feature_Extractor()

    # Get Feature Sets
    mFeature_Sets = Feature_DB()
    mFeature_Sets.load(default_feature_database_sets_path)

    # get query type.
    # 1)If query is a image,suffix should be the suffix of it;
    # 2)If query is a dir,suffix=''
    suffix = os.path.splitext(query)[1]

    #pdb.set_trace()

    #
    if suffix == '.jpg' or suffix == '.png' or suffix == '.jpeg':

        topN = 4
        result_class, result_image_path, result_conf = _search_by_image(mFeature_Extractor, mFeature_Sets, query, top=topN)


        print('\n')
        print('-----Show Top%d Results-----' % topN)
        for idx, data in enumerate(result_class):
            print('#%d class:%s, score:%f, path:%s' % (idx + 1, result_class[idx], result_conf[idx], result_image_path[idx]))
        print('\n')


        # plot result
        show_w = 400
        show_h = 300
        if should_plot_result:
            img_query = cv2.imread(query)
            img_query = cv2.resize(img_query, (show_w, show_h), interpolation=cv2.INTER_CUBIC)

            final_img = img_query

            for idx,path in enumerate(result_image_path):
                img = cv2.imread(path)
                img = cv2.resize(img, (show_w, show_h), interpolation=cv2.INTER_CUBIC)

                # put text
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (show_w/10, show_h/10)
                fontScale = 0.5
                fontColor = (255, 255, 255)
                lineType = 2
                cv2.putText(img, '%4f'%result_conf[idx],
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType)

                final_img = np.hstack((final_img, img))

            cv2.imshow('Top%d Results' % topN, final_img)

            k = cv2.waitKey(0)
            if k == ord('q'):
                cv2.destroyAllWindows()

            return 0

    else:
        print("Image format unsupported!")


# #############################
# Internal Function
# #############################
def _search_by_image(Extractor,FeatureSets,Query_Image_Path,conf_thres=0.001,top=10):
    """
    Searching by Image file

    :param Extractor:
    :param FeatureSets:
    :param Query_Image_Path:
    :param conf_thres:
    :param top:
    :return:
    """
    # --Get query image feature
    query_feat = Extractor.extract(Query_Image_Path)

    # --Get DB
    db = FeatureSets.Feat_Sets
    class_names = db['class']
    feats = db['feature']
    image_paths = db['path']

    # --Get scorer
    mScorer = Image_Scoring()

    # --list->np.array()
    query_feat = np.array([query_feat]).reshape(1,query_feat.shape[0])
    feats = np.array(feats)

    # --Scoring
    scores = mScorer.cosine_distance(query_feat,feats)[0]

    #pdb.set_trace()

    # --Sorting
    rank_ID = np.argsort(-scores)
    rank_score = scores[rank_ID]

    # --Get top ranking results
    result_class_name = []
    result_image_path = []
    result_score = []

    for k in range(0, top):

        if rank_score[k] > conf_thres:
            result_class_name.append(class_names[rank_ID[k]])
            result_image_path.append(image_paths[rank_ID[k]])
            result_score.append(rank_score[k])
            # print(rank_ID[k]+1,frame_infos[rank_ID[k]],rank_score[k])

    return result_class_name, result_image_path, result_score

if __name__ == '__main__':
    # Query Image Info
    query_img_name = 'frame_28.jpg'
    query_img_dir = '/home/tfl/workspace/dataSet/Visual_Instance_Retrieval/Rango-clip/images/'
    query_img_path = os.path.join(query_img_dir, query_img_name)

    content_base_image_retrieval(query_img_path)