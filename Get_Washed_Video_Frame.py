# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import cv2
import pdb

from conf import *

class DHash(object):
    @staticmethod
    def calculate_hash(image):
        """
        计算图片的dHash值
        :param image: PIL.Image
        :return: dHash值,string类型
        """
        difference = DHash.__difference(image)
        # 转化为16进制(每个差值为一个bit,每8bit转为一个16进制)
        decimal_value = 0
        hash_string = ""
        for index, value in enumerate(difference):
            if value:  # value为0, 不用计算, 程序优化
                decimal_value += value * (2 ** (index % 8))
            if index % 8 == 7:  # 每8位的结束
                hash_string += str(hex(decimal_value)[2:].rjust(2, "0"))  # 不足2位以0填充。0xf=>0x0f
                decimal_value = 0
        return hash_string

    @staticmethod
    def hamming_distance(first, second):
        """
        计算两张图片的汉明距离(基于dHash算法)
        :param first: Image或者dHash值(str)
        :param second: Image或者dHash值(str)
        :return: hamming distance. 值越大,说明两张图片差别越大,反之,则说明越相似
        """
        # A. dHash值计算汉明距离
        if isinstance(first, str):
            return DHash.__hamming_distance_with_hash(first, second)
        # B. image计算汉明距离
        hamming_distance = 0
        image1_difference = DHash.__difference(first)
        image2_difference = DHash.__difference(second)
        for index, img1_pix in enumerate(image1_difference):
            img2_pix = image2_difference[index]
            if img1_pix != img2_pix:
                hamming_distance += 1
        return hamming_distance

    @staticmethod
    def __difference(image):
        """
        *Private method*
        计算image的像素差值
        :param image: PIL.Image
        :return: 差值数组。0、1组成
    """
        resize_width = 9
        resize_height = 8
        # 1. resize to (9,8)
        smaller_image = image.resize((resize_width, resize_height))
        # 2. 灰度化 Grayscale
        grayscale_image = smaller_image.convert("L")
        # 3. 比较相邻像素
        pixels = list(grayscale_image.getdata())
        difference = []
        for row in range(resize_height):
            row_start_index = row * resize_width
            for col in range(resize_width - 1):
                left_pixel_index = row_start_index + col
                difference.append(pixels[left_pixel_index] > pixels[left_pixel_index + 1])
        return difference

    @staticmethod
    def __hamming_distance_with_hash(dhash1, dhash2):
        """
        *Private method*
        calculate hamming distance with dhash value
        :param dhash1: str
        :param dhash2: str
        :return: hamming_distance(int)
        """
        difference = (int(dhash1, 16)) ^ (int(dhash2, 16))
        return bin(difference).count("1")

# ####################
# API
# ####################
def get_key_frame_and_info(in_video_path,in_video_name,out_dir,dhash_dis_thres=15):
    if not os.path.isdir(os.path.join(out_dir, in_video_name)):
        # ./
        os.mkdir(os.path.join(out_dir, in_video_name))
        # ./images
        os.mkdir(os.path.join(out_dir, in_video_name, 'images'))

    # --Get video frames and Info
    cap = cv2.VideoCapture(in_video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # get FPS
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3:
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        print "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)

    # Get Key frame
    i = 0
    key_frame_cnt = 0
    while (cap.isOpened()):
        i = i + 1
        ret, frame = cap.read()
        if (ret == True):
            if (i == 1):
                image1 = frame
                print(type(image1))

            if (i > 1):
                image2 = frame
                # 只有在第二帧的时候给image1赋值，后面满足dhash_dis条件才给image1赋值，防止反复fromarray
                if (i == 2):
                    image1 = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
                image2 = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

                image1_dhash = DHash.calculate_hash(image1)
                image2_dhash = DHash.calculate_hash(image2)
                dhash_dis = DHash.hamming_distance(image1_dhash, image2_dhash)
                if (dhash_dis >= dhash_dis_thres):
                    key_frame_cnt = key_frame_cnt + 1
                    image1 = frame
                    image1 = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
                    print('Processing:', i, '  ', dhash_dis)

                    save_frame_path = os.path.join(out_dir, in_video_name, 'images',
                                                   'frame_' + str(i) + '.jpg')
                    # resize
                    frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_CUBIC)

                    cv2.imwrite(save_frame_path, frame)
        else:
            break
    cap.release()

    # Get video Info
    video_info_file_path = os.path.join(out_dir, in_video_name, 'video_info.txt')
    fid = open(video_info_file_path, 'w')

    fid.write('Name:%s\n' % in_video_name)
    fid.write('fps:%d\n' % fps)
    fid.write('frame_size:%dx%d\n' % (frame_width, frame_height))
    fid.write('frame_num:%d\n' % i)
    fid.write('key_frame_num:%d\n' % key_frame_cnt)

    fid.close()

# #################
# Main Process
# #################
if __name__ == '__main__':
    # --Dir
    demo_dir = os.path.join(wd, 'demo')
    save_dir = os.path.join(demo_dir, 'video_key_frames')

    # --init cv2-image object
    #warmingup_img_path = os.path.join(demo_dir, '2.jpg')
    #image1 = cv2.imread(warmingup_img_path)
    #image2 = cv2.imread(warmingup_img_path)

    # --get demo video
    demo_video_name = 'Rango-clip'
    demo_video_suffix = '.flv'
    demo_video_path = os.path.join(demo_dir, demo_video_name + demo_video_suffix)

    #
    get_key_frame_and_info(demo_video_path,demo_video_name,save_dir)


