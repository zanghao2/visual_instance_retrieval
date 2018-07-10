import os
import sys
import random
import shutil

imgdir = '/mnt/lvm/hdd/zhaoguangyu/data/missfresh_goods'
savedir = '/mnt/lvm/hdd/zhaoguangyu/data'
dirlist = os.listdir(imgdir)

search_p = open("./lib_name.txt",'wt')
query_p = open('./query_name.txt','wt')



for n,m in enumerate(dirlist):
    print n,m
    imgd = os.path.join(imgdir,m,'images')
    if not os.path.exists(imgd):
        print '%s not exist'%imgd
        continue
    print imgd
    imglist = os.listdir(imgd)
    imgnum = len(imglist)
    use_len = imgnum/10
    real_len = 0

    search_p_dir = os.path.join(savedir,'libimg',m,'images')
    if not os.path.exists(search_p_dir):
        os.makedirs(search_p_dir)
    query_p_dir = os.path.join(savedir,'queryimg',m,'images')
    if not os.path.exists(query_p_dir):
        os.makedirs(query_p_dir)

    for imgname in imglist:
        rval = random.random()
        if rval < 0.1:
            search_p.write('%s/%s/%s\n'%(m,'images',imgname))
            real_len += 1
            shutil.copyfile(os.path.join(imgd,imgname),os.path.join(search_p_dir,imgname))
        else:
            query_p.write('%s/%s/%s\n'%(m,'images',imgname))
            shutil.copyfile(os.path.join(imgd,imgname),os.path.join(query_p_dir,imgname))
    print 'search len %d , %d'%(real_len,use_len)
query_p.close()
search_p.close()

        
