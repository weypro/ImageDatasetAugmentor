import os
from typing import List
import numpy as np
import cv2
import re


from .utils import ImageFileUtil

class ImagePathLoader:

    path_list = []
    img_root_path = ""

    def __init__(self, path):
        self.set_root_path(path)
        self.load()
    
    def set_root_path(self, path):
        self.img_root_path = path

    def load(self):
        try:
            self.path_list = ImageFileUtil.get_images_path_list(self.img_root_path)
        except:
            print("Can't load images. Please check the path.")

    def content(self):
        return self.path_list
