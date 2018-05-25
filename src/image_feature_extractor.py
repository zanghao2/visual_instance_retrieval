# -*- coding: utf-8 -*-

import os,sys
import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import PCA
import pdb

os.environ['GLOG_minloglevel'] = '2'
import caffe

from conf import *

# --Caffe Model
caffe_model_path = os.path.join(wd, 'material', 'models', 'yi+shopping.caffemodel')
caffe_prototxt_path = os.path.join(wd, 'material', 'models', 'yi+shopping.prototxt')

if not os.path.exists(caffe_model_path):
    print('Caffe model file missing!')
    exit()

if not os.path.exists(caffe_prototxt_path):
    print('Caffe prototxt file missing!')
    exit()

# --ILSVRC2012 Mean file
ilsvrc2012_mean_npy_path = os.path.join(wd, 'material', 'models', 'ilsvrc_2012_mean.npy')
if not os.path.exists(ilsvrc2012_mean_npy_path):
    print('ilsvrc12 mean file missing!')
    exit()


class FeatureExtractor:
    """
    Extract Features from a input image.

    Using caffe model

    """
    def __init__(self, method='R-MAC', layer='inception_5b/output', pool=None, norm=None, MAX_BATCHSIZE=2):
        print('--Init Feature_Extractor')

        # load ilsvrc12 mean
        ilsvrc12_mean = np.load(ilsvrc2012_mean_npy_path)
        ilsvrc12_mean = ilsvrc12_mean.mean(1).mean(1)
        self.ilsvrc12_mean = ilsvrc12_mean

        # init caffe model
        net = caffe.Net(caffe_prototxt_path,  # defines the structure of the model
                        caffe_model_path,  # contains the trained weights
                        caffe.TEST)  # use test mode (e.g., don't perform dropout)
        self.net = net

        # init image transformer
        self.transformer = self._init_data_transformer()

        # feature extract method
        self.method = method

        # layer
        self.layer = layer

        # pooling
        self.pool = pool

        # normalization
        self.norm = norm

        # MAX_BATCHSIZE
        self.MAX_BATCHSIZE = MAX_BATCHSIZE

    def extract(self, image_path, method='R-MAC', layer='inception_5b/output', pool=None, norm=None):
        """
        Extract image Feature-Vector from off-the-shelf CNN Network.

        :param image_path: path of source image
        :param method:
                    R-MAC -- Regional Maximum Activation of Convolution.
                    MAC   -- Maximum Activation of Convolution,
                    Vec   -- Feature Vector
        :param layer:
        :param pool:
        :param norm:
        :return:

        """

        if not os.path.exists(image_path):
            print("Source image is invalid!")
            exit()

        print("#Get Image %s Features"%method)

        if method == 'R-MAC':
            if layer == '':
                print('You should specify a CNN layer for R-MAC')
                exit()
            return self._extract_RMAC_feature(image_path, layer=layer)
        elif method == 'MAC':
            if layer == '':
                print('You should specify a CNN layer for MAC')
            return self._extract_MAC_feature(image_path, layer=layer)

        elif method == 'Vec':
            if layer == '':
                print('You should specify a CNN layer for Vec')
            return self._extract_feature(image_path, layer=layer, pool=pool, norm=norm)

        else:
            print("Unknown Method!")
            exit()

    # TODO：支持不同大小的batchsize的输入。目前，还为用在具体的应用上。以后可以添加
    def extract_paths(self, image_path_list, layer='pool5/7x7_s1', norm='l2'):
        """
        Extract Features from a input image paths

        :param image_path_list: [path of input image]
        :param layer: define layer
        :param norm: should normalize the features
        :return:
        """
        feats = []

        # get number of input images
        imgNum = len(image_path_list)

        # get max batch_size
        max_bz = self.MAX_BATCHSIZE

        # get input images
        imgs = [caffe.io.load_image(path) for path in image_path_list]
        imgs = np.array(imgs)

        # get "cwh"
        dims = self.transformer.inputs['data'][1:]

        for chunk in [imgs[x:min((x + max_bz),len(imgs))] for x in xrange(0, len(imgs), max_bz)]:
            # get real batch size
            real_bz = len(chunk)
            new_shape = (real_bz,) + tuple(dims)

            # resize network
            if self.net.blobs['data'].data.shape != new_shape:
                self.net.blobs['data'].reshape(*new_shape)

            # fill Data_Layer
            for index, image in enumerate(chunk):
                transformed_image = self.transformer.preprocess('data', image)
                self.net.blobs['data'].data[index] = transformed_image

            # get features
            fts = self.net.forward()[layer]

            # TODO: only support L2-norm right now
            # Normalization
            if norm=='None':
                pass
            else:
                fts = [np.squeeze(ft) for ft in fts]
                fts = [ft / LA.norm(ft) for ft in fts]

            feats = feats+fts

        return feats

    # ###################
    # Internal
    # ###################
    def _init_data_transformer(self):
        """
        Init Data_Layer Transformer

        :return:
        """
        transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', self.ilsvrc12_mean)
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2, 1, 0))

        return transformer

    # -------------------- #
    # Feature Extractor
    # -------------------- #
    def _extract_feature(self, image_path, layer='pool5/7x7_s1', pool='avg', norm='l2'):
        """
        Extract Features Map/Vector from a source image

        :param image_path: path of input image
        :param layer: cnn layer to be extracted
        :param pool: support avg/max/sum pooling
        :param norm: should normalize the features

        :return:
                feature map/vector (chw)

        """

        # reshape
        if 1 != self.net.blobs['data'].data.shape[0]:
            self.net.blobs['data'].reshape(1, 3, 224, 224)

        # read image
        img = caffe.io.load_image(image_path)

        # transform
        transformed_image = self.transformer.preprocess('data', img)
        self.net.blobs['data'].data[...] = transformed_image

        # extract features
        self.net.forward()
        ft = self.net.blobs[layer].data[0]

        # pooling
        if pool == 'None':
            return ft

        ft = self._pooling_3D(ft, pool=pool)

        # TODO: only support L2-norm right now
        # Normalization
        if norm == 'None':
            return ft

        ft = self._normalize_array(ft, norm=norm)

        return ft

    def _extract_RMAC_feature(self, image_path, layer='inception_5b/output', region_lever=4):
        """
        Extracting the R-MAC feature of a image

        :param image_path: path of the input image
        :return: R-MAC feature
        """
        r_mac_list = []

        # get last cnn feature map(should be 1024x7x7)
        ft = self._extract_feature(image_path, layer=layer, pool='None')

        # get channel/height/width
        ft_c, ft_h, ft_w = ft.shape

        if ft_h < 3 and ft_w < 3:
            print("Too small feature map,no need to use R-MAC")
            exit()

        # get R-MAC feature
        for itr in range(region_lever):
            # region lever
            lv = itr+1
            # region size
            r_sz = 2*np.min([ft_h,ft_w])/(lv+1)

            # get each region
            for x in range(ft_w):
                for y in range(ft_h):
                    x_st = x
                    y_st = y
                    x_end = x + r_sz
                    y_end = y + r_sz

                    # check boundary
                    if x_end > ft_w or y_end > ft_h:
                        continue

                    # get MAC vector
                    region_feat_map = ft[:, y_st:y_end, x_st:x_end]
                    region_mac = self._MAC(region_feat_map)
                    # print("#lv:%d--%dx%dx%d"%(lv,region_feat_map.shape[0],region_feat_map.shape[1],region_feat_map.shape[2]))
                    r_mac_list.append(region_mac)

        # format change
        r_mac_array = np.array(r_mac_list)

        #pdb.set_trace()

        # l2-norm
        norm_r_mac_array = self._l2_norm_array(r_mac_array)

        # PCA-whitening
        whiten_r_mac_array = self._PCA_whiten(norm_r_mac_array)

        # l2-norm
        norm_whiten_r_mac_array = self._l2_norm_array(whiten_r_mac_array)

        # sum
        sum_r_mac_vec = np.sum(norm_whiten_r_mac_array,0)

        # l2-norm
        norm_sum_r_mac_vec = sum_r_mac_vec/LA.norm(sum_r_mac_vec)

        return norm_sum_r_mac_vec

    def _extract_MAC_feature(self, image_path, layer=''):

        # get last cnn feature map(should be 1024x7x7)
        ft = self._extract_feature(image_path, layer=layer, pool='None')

        return self._MAC(ft)

    def _MAC(self, feature_map_3D):
        """
        Calculate Maximum Activation of Convolution

        :param feature_map_3D: feature map or crop of feature map
        :return:
        """
        mac = []

        # chw of feature_map
        channel,height,width = feature_map_3D.shape

        if channel==0 or height == 0 or width == 0:
            print('Invilid Input data for _MAC()')
            exit()

        mac = self._max_pooling_3D(feature_map_3D)

        return mac

    # -------------------- #
    # Pooling
    # -------------------- #
    def _pooling_3D(self,feature_map_3D,pool='avg'):
        """
        Pooling

        :param feature_map_3D: feature map of c,h,w
        :param pool: avg/max/sum
        :return:
                feature vector
        """
        if pool == 'avg':
            return self._avg_pooling_3D(feature_map_3D)
        elif pool == 'max':
            return self._max_pooling_3D(feature_map_3D)
        elif pool == 'sum':
            return self._sum_pooling_3D(feature_map_3D)
        else:
            print("Invalid pooling method!")
            exit()

    def _max_pooling_3D(self, feature_map_3D):

        # chw of feature_map
        channel, height, width = feature_map_3D.shape

        if channel == 0 or height == 0 or width == 0:
            print('Invilid Input data for _MAC()')
            exit()

        return np.max(np.max(feature_map_3D, axis=1), axis=1)

    def _avg_pooling_3D(self, feature_map_3D):

        # chw of feature_map
        channel, height, width = feature_map_3D.shape

        if channel == 0 or height == 0 or width == 0:
            print('Invalid Input data for _MAC()')
            exit()

        return np.mean(np.mean(feature_map_3D, axis=1), axis=1)

    def _sum_pooling_3D(self, feature_map_3D):

        # chw of feature_map
        channel, height, width = feature_map_3D.shape

        if channel == 0 or height == 0 or width == 0:
            print('Invilid Input data for _MAC()')
            exit()

        return np.sum(np.sum(feature_map_3D, axis=1), axis=1)

    # -------------------- #
    # Normalization
    # -------------------- #
    def _normalize_array(self, array_n_x_m, norm='l2'):
        """
        Do normalization

        :param array_n_x_m:
        :param norm:
        :return:
        """
        if norm == 'l2':
            return self._l2_norm_array(array_n_x_m)
        else:
            print("Unsupported normalize methods!")
            exit()

    def _l2_norm_array(self, array_n_x_m):
        """
        L2 Norm

        :param array_n_x_m:
        :return:
        """
        # feature vector
        if len(array_n_x_m.shape) == 1:
            return array_n_x_m / LA.norm(array_n_x_m)
        # feature map
        for idx, vec in enumerate(array_n_x_m):
            vec = np.squeeze(array_n_x_m[idx])
            array_n_x_m[idx] = vec / LA.norm(vec)
        return array_n_x_m

    # -------------------- #
    # Whiten
    # -------------------- #
    def _PCA_whiten(self, array_n_x_m):
        """
        Do PCA-Whitening

        :param array_n_x_m:
        :return:
        """
        pca = PCA(whiten=True, copy=False)
        pca.fit(array_n_x_m)
        return array_n_x_m

if __name__ == '__main__':
    # Init Feature Extractor
    mFeature_Extractor = FeatureExtractor()

    # image paths
    img1_path = '/home/tfl/workspace/project/content_based_video_retrieval/demo/query_images/test01.jpg'
    img2_path = '/home/tfl/workspace/project/content_based_video_retrieval/demo/query_images/test02.jpg'
    img3_path = '/home/tfl/workspace/project/content_based_video_retrieval/demo/query_images/test03.jpg'

    paths = [img1_path, img2_path, img3_path]
    ft = mFeature_Extractor.extract(img1_path, method='Vec', layer='pool5/7x7_s1', pool='None', norm='l2')
    ft1 = mFeature_Extractor.extract(img1_path, method='R-MAC', layer='inception_5b/output', pool='None', norm='None')
    pdb.set_trace()

    exit()

    # path
    print('--single mode--')
    ft = mFeature_Extractor.extract(img1_path)
    print('img1:%s' % ft[0:10])
    ft = mFeature_Extractor.extract(img2_path)
    print('img2:%s' % ft[100:106])
    ft = mFeature_Extractor.extract(img3_path)
    print('img3:%s' % ft[500:510])

    # paths
    print('--batch mode--')
    fts = mFeature_Extractor.extract_paths(paths)
    pdb.set_trace()