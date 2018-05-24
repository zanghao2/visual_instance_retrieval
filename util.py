import numpy as np
import os
import cv2
import glob


# ################################
# Utils
# ################################
def get_file_name_in_dir(in_dir,out_path,suffix='*.jpg'):
    '''
    Get the file names in "in_dir"

    :param in_dir: file dir
    :param out_path: save path
    :param suffix: file suffix
    :return: None
    '''
    filelist = glob.glob(os.path.join(in_dir, suffix))
    #pdb.set_trace()
    f = open(out_path,'w')
    for line in filelist:
        name = line.split('/')[-1].split('.')[0]
        f.write(name+'\n')
    f.close()

    return 0

def get_file_full_path_in_dir(files_dir,out_path,suffix='*.jpg'):
    '''
    Get file full path from directory "files_dir"

    :param files_dir:
    :param out_path:
    :param suffix:
    :return:
    '''
    filelist = glob.glob(os.path.join(files_dir, suffix))
    f = open(out_path, 'w')
    for line in filelist:
        f.write(line + '\n')
    f.close()

def change_img_format(in_dir,out_dir,in_suffix='*.bmp'):
    #pdb.set_trace()
    filelist = glob.glob(os.path.join(in_dir, in_suffix))

    for imgpath in filelist:
        im = cv2.imread(imgpath)
        imgname = imgpath.split('/')[-1].split('.')[0]

        # remove "blankspace" in image names
        imgname = imgname.split(' ')
        imgname = ''.join(imgname)

        outpath = os.path.join(outdir,imgname+'.jpg')
        cv2.imwrite(outpath,im)
        #pdb.set_trace()

    pass

def sec_to_hms(sec):
    hour = int(sec / 3600)
    minute = int((sec - 3600 * hour) / 60)
    second = int((sec - 3600 * hour - 60 * minute))

    return hour,minute,second

# ################################
# Algorithm
# ################################



# ################################
# Test
# ################################
if __name__ == "__main__":

    # change images format
    imgdir = '/workspace/Share/data_transfer/video_key_frames/shajizhi/images'
    outdir = '/workspace/Share/data_transfer/video_key_frames/shajizhi/images'
    change_img_format(imgdir,outdir,'*.png')
