import os,sys
from image_feature_extractor import Feature_Extractor
import glob
import pdb
import cPickle

from conf import *

class Video_Feature_Sets:
    """
    Manage Video Feature Sets

    """

    Feat_Sets = {}

    Feat_Extractor = None

    def __init__(self):
        self.Feat_Extractor = Feature_Extractor()

    def load(self,path):
        if not os.path.exists(path):
            print('No such file! Make sure you get the right path!')
            exit()

        suffix = os.path.splitext(path)[1]
        if suffix != '.pkl':
            print('Feature Sets is not a pkl format file!')
            exit()

        self.Feat_Sets = cPickle.load(open(path,'rb'))


    def create(self,video_folder):
        """
        Creating a new video feature sets

        :param video_folder:
        :return:
        """
        frame_dir = os.path.join(video_folder,'images')
        video_info_file_path = os.path.join(video_folder,'video_info.txt')

        video_info_dict,frames_info,feats = self._get_video_features_and_info(frame_dir,video_info_file_path)

        # Creat Feat_Sets        
        self.Feat_Sets['video_info'] = [video_info_dict]
        self.Feat_Sets['frame_info'] = frames_info
        self.Feat_Sets['frame_feature'] = feats
        
        # dump to pickle file
        fid = open(default_video_feature_sets_path,'wb')
        cPickle.dump(self.Feat_Sets,fid)
        fid.close()


    def add(self,video_folder):
        """
        Add new video to current video feature sets

        :param video_folder:
        :return:
        """
        frame_dir = os.path.join(video_folder, 'images')
        video_info_file_path = os.path.join(video_folder, 'video_info.txt')

        if not os.path.exists(frame_dir):
            print('Video frame dir missing! Add failed!')
            exit()

        if not os.path.exists(video_info_file_path):
            print('Video info file missing! Add failed!')
            exit()

        # Load exist Feature Sets if there is one
        if len(self.Feat_Sets)==0:
            self.load(default_video_feature_sets_path)

        # Add new video features
        video_info_dict, frames_info, feats = self._get_video_features_and_info(frame_dir, video_info_file_path)

        # Creat Feat_Sets
        self.Feat_Sets['video_info'] = self.Feat_Sets['video_info']+[video_info_dict]
        self.Feat_Sets['frame_info'] = self.Feat_Sets['frame_info']+frames_info
        self.Feat_Sets['frame_feature'] = self.Feat_Sets['frame_feature']+feats

        # dump to pickle file
        fid = open(default_video_feature_sets_path,'wb')
        cPickle.dump(self.Feat_Sets,fid)
        fid.close()

        #pdb.set_trace()

    def remove(self):
        pass

    def _get_video_features_and_info(self,frame_dir,video_info_file_path):
        # get feature
        feats = []
        frame_paths_list = glob.glob(os.path.join(frame_dir, '*.jpg'))
        frame_paths_list.sort(key=lambda name: int(name.split('_')[-1].split('.')[0]))  # sort by frame NO.
        frame_names_list = [x.split('/')[-1].split('.')[0] for x in frame_paths_list]

        for path in frame_paths_list:
            print('Extract--%s'%path.split('/')[-1])
            ft = self.Feat_Extractor.extract(path)
            feats.append(ft)

        # get video info
        fid = open(video_info_file_path, 'r')
        video_info = fid.readlines()
        fid.close()

        video_info = [x.strip() for x in video_info]

        video_info_dict = {}
        for line in video_info:
            k = line.split(':')[0]
            v = line.split(':')[1]
            video_info_dict[k] = v

        # get frame info
        frames_info = []
        for f_name in frame_names_list:
            v_name = video_info_dict['Name']
            v_name = v_name.replace(' ', '') # remove ' ' in name
            v_name = v_name.replace('_', '') # remobe '_' in name
            new_name = v_name + '_' + f_name
            frames_info.append(new_name)

        return video_info_dict,frames_info,feats


if __name__ == '__main__':

    my_Video_Feature_Sets = Video_Feature_Sets()

    # create a video feature sets if there is none
    if not os.path.exists(default_video_feature_sets_path):
        default_video_folder = os.path.join(video_key_frame_root,video_name)
        my_Video_Feature_Sets.create(default_video_folder)

    # add new video
    #video_folder = os.path.join(wd,'demo','video_key_frames','Rango-clip')
    #my_Video_Feature_Sets.add(video_folder)
