import cPickle
import glob

from conf import *
from src.image_feature_extractor import FeatureExtractor


class FeatureDB:
    """
    Manage Image Feature Sets

    """

    Feat_Sets = {}

    Feat_Extractor = None

    def __init__(self):
        self.Feat_Extractor = FeatureExtractor()

    def load(self, path):
        """
        Load existing FeatureDB if there is one

        :param path:
        :return:
        """
        if not os.path.exists(path):
            print('No such file! Make sure you get the right path!')
            exit()

        suffix = os.path.splitext(path)[1]
        if suffix != '.pkl':
            print('Feature Sets is not a pkl format file!')
            exit()

        self.Feat_Sets = cPickle.load(open(path,'rb'))

    def create(self, class_name, source_img_folder):
        """
        Creating a new image featureDB

        :param class_name: new class name
        :param source_img_folder:
        :return:
        """
        frame_dir = os.path.join(source_img_folder, 'images')

        feats, image_paths = self._get_image_feature_and_info(frame_dir)

        # get class name list
        class_name_list = [class_name for _ in feats]

        # Creat Feat_Sets        
        self.Feat_Sets['class'] = class_name_list
        self.Feat_Sets['path'] = image_paths
        self.Feat_Sets['feature'] = feats
        
        # dump to pickle file
        fid = open(default_feature_database_sets_path, 'wb')
        cPickle.dump(self.Feat_Sets,fid)
        fid.close()

    def add(self, class_name, source_img_folder):
        """
        Add new/exist class images to current image feature sets

        :param class_name: new/exist class name
        :param source_img_folder: dir
        :return:
        """
        frame_dir = os.path.join(source_img_folder, 'images')

        if not os.path.exists(frame_dir):
            print('Image file missing! Add failed!')
            exit()

        # Load exist Feature Sets if there is one
        if len(self.Feat_Sets)==0:
            self.load(default_feature_database_sets_path)

        # Add new video features
        feats, image_paths = self._get_image_feature_and_info(frame_dir)

        # get class name list
        class_name_list = [class_name for _ in feats]

        # Creat Feat_Sets
        self.Feat_Sets['class'] = self.Feat_Sets['class'] + class_name_list
        self.Feat_Sets['path'] = self.Feat_Sets['path'] + image_paths
        self.Feat_Sets['feature'] = self.Feat_Sets['feature'] + feats

        # dump to pickle file
        fid = open(default_feature_database_sets_path, 'wb')
        cPickle.dump(self.Feat_Sets,fid)
        fid.close()

        #pdb.set_trace()

    def remove(self, class_name):
        pass

    # ###########################
    # Internal functions
    def _get_image_feature_and_info(self, images_dir, method='R-MAC',
                                    layer='inception_5b/output', pool='None', norm='None'):

        feats = []

        frame_paths_list = glob.glob(os.path.join(images_dir, '*.jpg'))

        for idx,path in enumerate(frame_paths_list):
            print('#%d-Extract--%s' % (idx, path.split('/')[-1]))
            ft = self.Feat_Extractor.extract(path, method=method, layer=layer, pool=pool, norm=norm)
            feats.append(ft)

        return feats, frame_paths_list

    def _get_video_features_and_info(self, frame_dir, video_info_file_path):
        # get feature
        feats = []
        frame_paths_list = glob.glob(os.path.join(frame_dir, '*.jpg'))
        frame_paths_list.sort(key=lambda name: int(name.split('_')[-1].split('.')[0]))  # sort by frame NO.
        frame_names_list = [x.split('/')[-1].split('.')[0] for x in frame_paths_list]

        for idx,path in enumerate(frame_paths_list):
            print('#%d-Extract--%s'%(idx,path.split('/')[-1]))
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

    my_Feature_Sets = FeatureDB()

    # create a video feature sets if there is none
    class_name = 'Rango-clip'

    if not os.path.exists(default_feature_database_sets_path):
        default_video_folder = os.path.join(dataset_dir, class_name)
        my_Feature_Sets.create(class_name, default_video_folder)

    # add new video
    #video_folder = os.path.join(wd,'demo','video_key_frames','Rango-clip')
    #my_Video_Feature_Sets.add(video_folder)
