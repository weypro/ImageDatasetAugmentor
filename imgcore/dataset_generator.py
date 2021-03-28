import time
import numpy as np
import cv2
import os
import sys

from utils.utils import ImageFileUtil, ProgressBarUtil
from utils.image_path_loader import ImagePathLoader

from imgcore.operation_module import Sequential, Blur, PolygonClipping, \
 Resize, SaltPepperNoise, Rotate, RandomRotate, ImgSave, Shift, GaussianNoise, \
 RectClipping

class TestDatasetGenerator():
    src_path = None
    dest_path = None

    def __init__(self,
                 src_path: str,
                 dest_path: str) -> None:
        super().__init__()
        self.src_path = src_path
        self.dest_path = dest_path


    def execute(self):
        img_loader = ImagePathLoader(self.src_path)

        start_time = time.time()

        count = 0
        output_count = 0
        total_len = len(img_loader.content())
        for img_path in img_loader.content():
            per_start_time = time.time()

            img=ImageFileUtil.open(img_path)

            process_module1 = Sequential(Resize(564, 360),
                            RandomRotate(10, 10)
                            )

            process_module2 = Sequential(Blur(),
                            SaltPepperNoise(0.001),
                            # GaussianNoise(0,10),
                            Shift(10, 20),
                            RectClipping(10,10,200,200)
                            )

            process_module3 = Sequential(PolygonClipping(np.array([[10, 10], [15, 0], [35, 8], [100, 20], [300, 45], [280, 100], [350, 230], [30, 200]])))

            processed_img1 = process_module1(img)
            processed_img2 = process_module2(processed_img1)

            save_module = ImgSave(self.dest_path, ImageFileUtil.get_file_name(img_path)+ "_")
            save_module(processed_img1)
            save_module(processed_img2)

            count = count + 1
            output_count = output_count + 2

            per_end_time = time.time()
            ProgressBarUtil.update(count, total_len, per_end_time - per_start_time)

        end_time = time.time()

        dest_img_loader = ImagePathLoader(self.dest_path)

        print('\n\n%s Images generated in %s sec' % (output_count, round(end_time - start_time, 3)))
        