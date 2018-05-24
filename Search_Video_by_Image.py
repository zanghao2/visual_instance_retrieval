# -*- coding: utf-8 -*-
import numpy as np
import os,sys
import glob
import cv2

os.environ['GLOG_minloglevel'] = '2'
import caffe
from image_feature_extractor import Feature_Extractor
from Make_Video_Feature_Sets import Video_Feature_Sets
from Get_Washed_Video_Frame import get_key_frame_and_info
from util import sec_to_hms

import pdb

from conf import *

# set gpu mode
caffe.set_mode_gpu()

output = []
res_feats = []

# #############################
# API
# #############################
def content_base_video_retrieval(query,mashup=False,should_plot_result = False):
    """
    Main function of CBVR

    :param query: Query,should be a 'path' or 'dir'
    :return:
    """

    # Init Feature Extractor
    mFeature_Extractor = Feature_Extractor()

    # Get Feature Sets
    mVideo_Feature_Sets = Video_Feature_Sets()
    mVideo_Feature_Sets.load(default_video_feature_sets_path)

    # get query type.
    # 1)If query is a image,suffix should be the suffix of it;
    # 2)If query is a dir,suffix=''
    suffix = os.path.splitext(query)[1]
    
    #
    if suffix == ('.jpg' or '.png'):

        topN = 8
        result_frame_info,_ = _video_search_by_image(mFeature_Extractor, mVideo_Feature_Sets, query,top=topN)
        res_list = []

        # --Get db video info
        db_video_info = _get_db_video_info(mVideo_Feature_Sets)

        for data in result_frame_info:
            # Matched frame index
            tmp_data = data.split('_')
            db_video_name = tmp_data[0]
            db_video_idx = int(tmp_data[-1])
            # Matched frame time(unit sec)
            #pdb.set_trace()
            db_video_fps = int(db_video_info[db_video_name]['fps'])
            sec = (db_video_idx*1.0)/(db_video_fps*1.0)
            db_video_sec = int(np.floor(sec))
            db_video_h,db_video_m,db_video_s = sec_to_hms(db_video_sec)

            res_list.append([db_video_name,[db_video_h,db_video_m,db_video_s]])

        print('\n')
        print('-----Show Top%d Results-----' % topN)
        for idx,data in enumerate(res_list):
            print('#%d Name:%s, %d:%d:%d'%(idx+1,data[0],data[1][0],data[1][1],data[1][2]))
        print('\n')

        # plot result
        show_w = 400
        show_h = 300
        if should_plot_result:
            img_query = cv2.imread(query)
            img_query = cv2.resize(img_query,(show_w,show_h),interpolation=cv2.INTER_CUBIC)

            final_img = img_query

            for frame_info in result_frame_info:
                # get DB source image dir
                db_vname = frame_info.split('_')[0]
                db_image_dir = os.path.join(wd, 'demo', 'video_key_frames', db_vname, 'images')

                db_fname = 'frame_'+frame_info.split('_')[-1]+'.jpg'
                img = cv2.imread(os.path.join(db_image_dir,db_fname))
                img = cv2.resize(img, (show_w, show_h), interpolation=cv2.INTER_CUBIC)
                final_img = np.hstack((final_img,img))

            cv2.imshow('Top%d Results'%topN,final_img)

            k = cv2.waitKey(0)
            if k == ord('q'):
                cv2.destroyAllWindows()

            return 0

    else:
        # ------------------
        # Mashup Video
        # ------------------
        if mashup:
            mashup_match_results = _video_search_by_mashup_video(mFeature_Extractor, mVideo_Feature_Sets, query)

            if len(mashup_match_results) == 0:
                print('\n')
                print('-----Results-----')
                print('No Matching Videos...')
                print('\n')

                return 0

            print('\n')
            print('-----Results-----')

            for idx,res in enumerate(mashup_match_results):
                vn = res[0]
                v_st_h, v_st_m, v_st_s = sec_to_hms(res[1])
                v_end_h, v_end_m, v_end_s = sec_to_hms(res[2])
                q_st_h, q_st_m, q_st_s = sec_to_hms(res[3])
                q_end_h, q_end_m, q_end_s = sec_to_hms(res[4])
                print('#%d  Video Name:%s, from %d:%d:%d to %d:%d:%d; Query: from %d:%d:%d to %d:%d:%d ' % (idx, vn, v_st_h, v_st_m, v_st_s, v_end_h, v_end_m, v_end_s, q_st_h, q_st_m, q_st_s, q_end_h, q_end_m, q_end_s))

            print('\n')

            return 0

        # ------------------
        # Normal Video
        # ------------------
        v_name,v_st_sec,v_end_sec = _video_search_by_video_clip(mFeature_Extractor, mVideo_Feature_Sets, query)

        if v_name == None:
            print('\n')
            print('-----Results-----')
            print('No Matching Videos...')
            print('\n')

            return 0

        v_st_h,v_st_m,v_st_s = sec_to_hms(v_st_sec)
        v_end_h, v_end_m, v_end_s = sec_to_hms(v_end_sec)

        print('\n')
        print('-----Results-----')
        print('Video Name:%s, from %d:%d:%d to %d:%d:%d' % (v_name, v_st_h,v_st_m,v_st_s, v_end_h, v_end_m, v_end_s))
        print('\n')

        return 0

# #############################
# Internal Function
# #############################
def _video_search_by_image(Extractor,FeatureSets,Query_Image_Path,conf_thres=0.5,top=10):
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
    ft = Extractor.extract(Query_Image_Path)

    # --Get DB
    db = FeatureSets.Feat_Sets
    frame_feats = db['frame_feature']
    frame_infos = db['frame_info']

    # --list->np.array()
    ft = np.array(ft)
    frame_feats = np.array(frame_feats)

    # --Matching
    scores = np.dot(ft, frame_feats.T)
    scores = np.array(scores)
    rank_ID = np.argsort(-scores)
    rank_score = scores[rank_ID]

    # --Get top ranking results
    result_frame_info = []
    result_score = []

    for k in range(0,top):

        if rank_score[k]>conf_thres:
            result_frame_info.append(frame_infos[rank_ID[k]])
            result_score.append(rank_score[k])
            #print(rank_ID[k]+1,frame_infos[rank_ID[k]],rank_score[k])

    return result_frame_info,result_score

def _video_search_by_video_clip(Extractor,FeatureSets,Query_Image_dir,conf_thres=0.5,top=10):
    """
    Searching by video-clip

    :param Extractor:
    :param FeatureSets:
    :param Query_Image_dir:
    :param conf_thres:
    :param top:
    :return:
    """

    # --number of matched frames should be larger than this value
    MATCH_FRAMES_NUM_THRES = 10

    if not os.path.isdir(Query_Image_dir):
        print('Invalid Query Images Dir ...')
        exit()

    # --Get db video info
    db_video_info = _get_db_video_info(FeatureSets)

    # --Get Washed Video Frames and info of Query Video clip
    query_frames_dir = os.path.join(Query_Image_dir,'images')
    query_info_path = os.path.join(Query_Image_dir,'video_info.txt')

    # get query frames abspath
    key_frame_list = glob.glob(os.path.join(query_frames_dir,'*.jpg'))
    key_frame_list.sort(key=lambda name: int(name.split('_')[-1].split('.')[0])) #sort by frame NO.

    # get query info
    fid = open(query_info_path, 'r')
    query_info = fid.readlines()
    fid.close()

    query_info = [x.strip() for x in query_info]

    query_info_dict = {}
    for line in query_info:
        k = line.split(':')[0]
        v = line.split(':')[1]
        query_info_dict[k] = v

    query_fps = int(query_info_dict['fps'])

    # --Searching
    query_frame_sec_list = []
    res_dict = {}

    for frame_path in key_frame_list:
        # query frame index
        query_frame_idx = frame_path.split('/')[-1].split('.')[0].split('_')[-1]
        query_frame_idx = int(query_frame_idx)

        # query frame time(unit sec)
        sec = (query_frame_idx * 1.0) / (query_fps * 1.0)
        query_frame_sec = int(np.floor(sec))

        query_frame_sec_list.append(query_frame_sec)

        # Matching query image with DB
        res,_ = _video_search_by_image(Extractor, FeatureSets, frame_path,top=50)

        #pdb.set_trace()

        # static one frame result
        for data in res:
            # Matched frame index
            tmp_data = data.split('_')
            db_video_name = tmp_data[0]
            db_video_idx = int(tmp_data[-1])
            # Matched frame time(unit sec)
            db_video_fps = int(db_video_info[db_video_name]['fps'])
            sec = (db_video_idx*1.0)/(db_video_fps*1.0)
            db_video_sec = int(np.floor(sec))

            # time diff between 'Query' and 'Matching frames'
            diff_sec = db_video_sec-query_frame_sec

            if res_dict.has_key(db_video_name):
                res_dict[db_video_name].append(diff_sec)
            else:
                res_dict[db_video_name]=[diff_sec]

    # --Get Query video clip duration(unit sec)
    query_dur_sec = query_frame_sec_list[-1]-query_frame_sec_list[0]

    # --Find max matching time-diff for all video in DB
    max_cnt = 0
    max_diff_sec = -1
    max_vname = None
    for key,val in res_dict.iteritems():
        val_unique = list(set(val))

        for data in val_unique:
            cnt = val.count(data)
            if cnt < MATCH_FRAMES_NUM_THRES:
                continue
            if cnt > max_cnt:
                max_cnt = cnt
                max_diff_sec = data
                max_vname = key

    # --Get final results
    if max_cnt<MATCH_FRAMES_NUM_THRES:
        return None,None,None

    match_db_video_name = max_vname
    match_db_sec_diff = max_diff_sec

    video_st = max(0, query_frame_sec_list[0] + match_db_sec_diff)
    video_end = video_st + query_dur_sec

    # print('video name:%s, start:%s, end:%s'%(match_db_video_name,video_st,video_end))

    return match_db_video_name, video_st, video_end


def _video_search_by_mashup_video(Extractor,FeatureSets,Query_Image_dir,conf_thres=0.6,top=10):
    """
    Searching by Mashup-video

    :param Extractor:
    :param FeatureSets:
    :param Query_Image_dir:
    :param conf_thres:
    :param top:
    :return:
    """
    # --Especially for Mashup-videos
    MATCH_FRAMES_NUM_THRES = 10

    if not os.path.isdir(Query_Image_dir):
        print('Invalid Query Images Dir ...')
        exit()

    # --Get db video info
    db_video_info = _get_db_video_info(FeatureSets)

    # --Get Washed Video Frames and info of Query Video clip
    query_frames_dir = os.path.join(Query_Image_dir,'images')
    query_info_path = os.path.join(Query_Image_dir,'video_info.txt')

    # get query frames abspath
    key_frame_list = glob.glob(os.path.join(query_frames_dir,'*.jpg'))
    key_frame_list.sort(key=lambda name: int(name.split('_')[-1].split('.')[0])) #sort by frame NO.

    # get query info
    fid = open(query_info_path, 'r')
    query_info = fid.readlines()
    fid.close()

    query_info = [x.strip() for x in query_info]

    query_info_dict = {}
    for line in query_info:
        k = line.split(':')[0]
        v = line.split(':')[1]
        query_info_dict[k] = v

    query_fps = int(query_info_dict['fps'])

    # --Searching
    query_frame_sec_list = []
    res_dict = {}
    time_diff_frame_time_dict = {}

    # TODO:暂时解决不了来自不同视频的混剪视频
    top_score_diff = []
    top_score_query_time = []
    top_score_vname = []

    for frame_path in key_frame_list:
        # query frame index
        query_frame_idx = frame_path.split('/')[-1].split('.')[0].split('_')[-1]
        query_frame_idx = int(query_frame_idx)

        # query frame time(unit sec)
        sec = (query_frame_idx * 1.0) / (query_fps * 1.0)
        query_frame_sec = int(np.floor(sec))

        query_frame_sec_list.append(query_frame_sec)

        # Matching query image with DB
        res,_ = _video_search_by_image(Extractor, FeatureSets, frame_path,conf_thres=conf_thres,top=top)

        #pdb.set_trace()

        # static one frame result
        for idx,data in enumerate(res):
            # Matched frame index
            tmp_data = data.split('_')
            db_video_name = tmp_data[0]
            db_video_idx = int(tmp_data[-1])
            # Matched frame time(unit sec)
            db_video_fps = int(db_video_info[db_video_name]['fps'])
            sec = (db_video_idx*1.0)/(db_video_fps*1.0)
            db_video_sec = int(np.floor(sec))

            # time diff between 'Query' and 'Matching frames'
            diff_sec = db_video_sec-query_frame_sec

            #
            if res_dict.has_key(db_video_name):
                res_dict[db_video_name].append(diff_sec)
            else:
                res_dict[db_video_name]=[diff_sec]

            # time-diff->[frame time]
            if time_diff_frame_time_dict.has_key(db_video_name):
                if time_diff_frame_time_dict[db_video_name].has_key(diff_sec):
                    time_diff_frame_time_dict[db_video_name][diff_sec].append(query_frame_sec)
                else:
                    time_diff_frame_time_dict[db_video_name][diff_sec] = [query_frame_sec]
            else:
                time_diff_frame_time_dict[db_video_name] = {diff_sec:[query_frame_sec]}

            if idx==0:
                top_score_diff.append(diff_sec)
                top_score_query_time.append(query_frame_sec)
                top_score_vname.append(db_video_name)

    # --Reconstruct top-score data
    top_score_dict = {}
    top_score_vname_unique = list(set(top_score_vname))
    for name in top_score_vname_unique:
        t_s_diff = []
        t_s_q_t = []
        for idx,vn in enumerate(top_score_vname):
            if vn==name:
                t_s_diff.append(top_score_diff[idx])
                t_s_q_t.append(top_score_query_time[idx])

        # smooth time-diff
        tmp_diff = -1
        diff_thres = 5  # unit second
        for idx, data in enumerate(t_s_diff):
            # first data
            if idx == 0:
                tmp_diff = data
                continue

            if abs(data - tmp_diff) < diff_thres:
                t_s_diff[idx] = tmp_diff
            else:
                tmp_diff = data

        top_score_dict[name]={'top_score_diff':t_s_diff,'top_score_query_time':t_s_q_t}

    # --Get Matching video clips
    match_clip_num = 0
    match_clip_results = []
    for vn,data in top_score_dict.iteritems():
        t_s_diff = data['top_score_diff']
        t_s_q_t = data['top_score_query_time']

        t_s_diff_unique = list(set(t_s_diff))
        for df in t_s_diff_unique:
            if t_s_diff.count(df)<MATCH_FRAMES_NUM_THRES:
                continue
            t_s_diff_array = np.array(t_s_diff)
            index = np.where(t_s_diff_array==df)[0]
            st = index[0]
            ed = index[-1]
            q_st = t_s_q_t[st]
            q_ed = t_s_q_t[ed]

            video_st = max(0, q_st + df)
            video_end = q_ed + df

            one_query_result = [vn,video_st,video_end,q_st,q_ed]
            match_clip_num = match_clip_num + 1

            match_clip_results.append(one_query_result)

            #pdb.set_trace()


        '''
        tmp_list = t_s_diff[1:]+[t_s_diff[-1]]
        diff_t_s_diff = np.array(tmp_list)-np.array(t_s_diff)
        diff_t_s_diff[0] = 1
        diff_t_s_diff[-1] = 1
        diff_idx = np.where(diff_t_s_diff>0)[0]

        for idx,pos in enumerate(diff_idx):
            if idx==0:
                continue
            st = diff_idx[idx-1]
            ed = pos
            
            match_clip_dict[vn]
        '''
    #pdb.set_trace()

    if match_clip_num==0:
        return []
    else:
        return match_clip_results


def _get_db_video_info(FeatureSets):
    """

    :param FeatureSets:
    :return:
    """
    video_info = {}

    db_video_info = FeatureSets.Feat_Sets['video_info']
    for info in db_video_info:
        v_name = info['Name']
        video_info[v_name] = info

    return video_info



if __name__ == '__main__':
    '''
    # Query Image Info
    query_img_name = 'frame_010899.jpg'
    query_img_dir = '/workspace/Share/data_transfer/video_key_frames/Game_of_Throne/images'
    query_img_path = os.path.join(query_img_dir,query_img_name)

    content_base_video_retrieval(query_img_path)
    '''
    # Query by video
    query_video_dir = wd+'/demo/video_key_frames/Rango-clip-408-435/'

    content_base_video_retrieval(query_video_dir)




