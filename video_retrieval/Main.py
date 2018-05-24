# -*- coding: utf-8 -*-
import time

from Make_Image_FeatureDB import Feature_DB
from Search_Video_by_Image import content_base_video_retrieval
from conf import *
from util import sec_to_hms
from video_retrieval.Get_Washed_Video_Frame import get_key_frame_and_info


# #####
# 清洗源视频
def Video_Washing():
    # --Dir
    demo_dir = os.path.join(wd, 'demo')
    save_dir = os.path.join(demo_dir, 'video_key_frames')

    # --get demo video
    demo_video_name = '5101060162_5001622515_92'
    demo_video_suffix = '.mp4'
    demo_video_path = os.path.join(demo_dir, demo_video_name + demo_video_suffix)

    #
    get_key_frame_and_info(demo_video_path, demo_video_name, save_dir)
    pass


# #####
# 生成视频库
def Make_DB():
    my_Video_Feature_Sets = Feature_DB()

    # create a video feature sets if there is none
    if not os.path.exists(default_feature_database_sets_path):
        default_video_folder = os.path.join(video_key_frame_root, video_name)
        my_Video_Feature_Sets.create(default_video_folder)

    # add new video
    print("# Adding...")
    video_folder = os.path.join(wd,'demo','video_key_frames','5101060162-5001622515-92')
    print('#%s'%video_folder)
    my_Video_Feature_Sets.add(video_folder)


# #####
# 视频检索
def Video_Retrieval():
    '''
    #
    should_make_new_data=False
    if should_make_new_data:
        # --Dir
        demo_dir = os.path.join(wd, 'demo')
        save_dir = os.path.join(demo_dir, 'video_key_frames')

        # --get demo video
        demo_video_name = 'battle01'
        demo_video_suffix = '.flv'
        demo_video_path = os.path.join(demo_dir, demo_video_name + demo_video_suffix)

        #
        get_key_frame_and_info(demo_video_path, demo_video_name, save_dir)
        pass

    #pdb.set_trace()
    '''

    # 以视频搜视频
    #Should_Plot_Result = False # 是否需要显示匹配结果(只在以图搜视频能用)
    #Is_Mashup_Video = True  # 是否为混编视频
    #query_dir = wd + '/demo/video_key_frames/Produce01/'

    # 以图搜视频
    Should_Plot_Result = True # 是否需要显示匹配结果(只在以图搜视频能用)
    Is_Mashup_Video = False
    query_dir = wd + '/demo/query_images/test09.jpg'

    content_base_video_retrieval(query_dir,Is_Mashup_Video,Should_Plot_Result)


if __name__ == '__main__':

    # ###############
    # --start-time
    print('# ----- %s'%time.ctime())
    stime = time.time()

    # ---------------

    # --Video_Washing
    #Video_Washing()

    # --Make_DB
    #Make_DB()

    # --Video_Retrieval
    Video_Retrieval()

    # ---------------

    # ###############
    # --end-time
    print('# ----- %s' % time.ctime())
    etime = time.time()

    dur_sec = etime - stime

    hour, minute, second = sec_to_hms(dur_sec)

    print ('Total processing time: %d h, %d m, %d s' % (hour, minute, second))