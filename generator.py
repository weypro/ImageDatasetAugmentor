#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
 @file    		    generator
 @brief    		    数据增强生成器
 @version    		V1.0
 @Software 		    Python 3.8
 @date       		2021-03
"""

import numpy as np
import cv2
import os
import sys
from matplotlib import pyplot as plt
from io import BytesIO
import time

from utils.utils import ImageFileUtil, ProgressBarUtil
from utils.image_path_loader import ImagePathLoader

from imgcore.dataset_generator import TestDatasetGenerator

from imgcore.operation_module import Sequential, Blur, PolygonClipping, \
 Resize, SaltPepperNoise, Rotate, RandomRotate, ImgSave, Shift, GaussianNoise, \
 RectClipping

# 获取根目录
ROOT_PATH = os.path.dirname(__file__)
REAL_ROOT_PATH = os.getcwd()
# 输入路径，要确保目录存在
inputimg_path = r'{}\imgs'.format(REAL_ROOT_PATH)
# 输出路径，要确保目录存在
outputimg_path = r'{}\outputimgs'.format(REAL_ROOT_PATH)


def main():
    img_generaetor = TestDatasetGenerator(inputimg_path, outputimg_path)
    img_generaetor.execute()

    


if __name__ == "__main__":
    main()

    #按任意键继续，如果想让窗口直接关闭注释掉就好
    # os.system('pause')
