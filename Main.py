# -*- coding: utf-8 -*-
from image_retrieval.Image_Retrieval import *

# ####################
# 以图搜图

# Query Image Info
query_img_name = '44.jpeg'
query_img_dir = '/home/tfl/workspace/dataSet/Visual_Instance_Retrieval/query_images/'
query_img_path = os.path.join(query_img_dir, query_img_name)

content_base_image_retrieval(query_img_path,should_plot_result=True)