import os
import shutil
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List

import numpy as np
from numpy.core.multiarray import ndarray
import cv2
import re


class StringUtil:
    @staticmethod
    def underScoreAndLowercase(words: str) -> str:
        return words.lower().replace(" ", "_")


class ProgressBarUtil:
    """
    Print the progress
    """

    @staticmethod
    def update(progress: int, total: int, rate: Optional[int] = None):
        workdone = progress / total
        print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)
        if rate is not None:
            print("    rate: {0:.2f}".format(rate), end="", flush=True)


class SortPathUtil:
    """Provide sorting methods

    It is used in ImageFileUtil.get_images_path_list
    """

    @staticmethod
    def sort_only_digit(s):
        return int(ImageFileUtil.get_file_name(s))

    @staticmethod
    def sort_letter_digit(s):
        """
        Sort the list when the elements contain both letters and dights
        """
        file_name = ImageFileUtil.get_file_name(s)
        
        # Calculate the 1st order
        # If the first character of the name is a letter, the order is its ASCII value
        # Or it will be 0 if it's a digit
        first_order = 0
        if file_name[0].isalpha():
            first_order = ord(file_name[0])

        # Calculate the 2nd order
        # Exact the number from the name as the second order
        # If it doesn't contain a number, the order will be 0
        try:
            second_order = int(re.search(r'\d+', file_name).group())
        except:
            second_order = 0
        
        # Combine them as a tuple and return
        return (first_order, second_order)


class ImageFileUtil:
    """
    A tool to manage the image file and directory
    """

    imageExtensions = ['bmp', 'jpeg', 'jpg', 'png', 'tiff']

    @staticmethod
    def folder_total_size(folder_path: str) -> float:
        return sum([os.path.getsize(os.path.join(folder_path, f))
                    for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))
                    and ImageFileUtil.is_image(os.path.join(folder_path, f))])

    @staticmethod
    def mean_folder_file_size(folder_path: str) -> float:
        return ImageFileUtil.folder_total_size(folder_path) / ImageFileUtil.get_image_counts_in_folder(folder_path)

    @staticmethod
    def get_image_counts_in_folder(folder_path: str) -> int:
        num_files = len(ImageFileUtil.get_images_path_list(folder_path))
        return num_files

    @staticmethod
    def get_file_extension(path: str) -> str:
        return os.path.basename(path).split('.')[1]

    @staticmethod
    def get_file_name(path: str) -> str:
        return os.path.basename(path).split('.')[0]

    @staticmethod
    def is_image(path: str) -> bool:
        return ImageFileUtil.get_file_extension(path).lower() in ImageFileUtil.imageExtensions

    @staticmethod
    def get_images_path_list(folder_path) -> List[str]:
        """
        Get the sorted list containing the full path of the images in the folder
        """
        img_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                os.path.isfile(os.path.join(folder_path, f))
                and ImageFileUtil.is_image(os.path.join(folder_path, f))]
        # If we doesn't sort it, it will be "1,10,11,2..."
        img_list.sort(key=SortPathUtil.sort_letter_digit)

        return img_list

    @staticmethod
    def open(path: str) -> ndarray:
        cvimg = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
        return img

    @staticmethod
    def create_folder(folder_path: str):
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

    @staticmethod
    def generate_next_file_path(folder_path: str, file_prefix: str, file_extension: str = ".png"):
        counter = len([i for i in os.listdir(folder_path) if file_prefix in i]) + 1
        file_name = file_prefix + str(counter) + file_extension
        return os.path.join(folder_path, file_name)

    @staticmethod
    def save_file(processed_image: ndarray, path: str, file_prefix: Optional[str] = ""):
        try:
            # If the path contains the file name, regard it as the destination path
            ImageFileUtil.get_file_extension(path)
            destination_path = path
        except:
            # If it doesn't contain the file name
            # Try to create a new folder
            ImageFileUtil.create_folder(path)
            # Concatenate it with a new file name to generate the destination path
            destination_path = ImageFileUtil.generate_next_file_path(path, file_prefix)

        # The image should be converted from RGB to BGR
        cv2.imwrite(destination_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))


    @staticmethod
    def rename_file_by_order(src_path: str, dest_path: str, file_prefix: Optional[str] = ""):
        """Copy the files to the new path and rename them by order
        
        The format is "prefix"+"id"

        """
        ImageFileUtil.create_folder(dest_path)
        img_list = ImageFileUtil.get_images_path_list(src_path)

        for img_id in range(len(img_list)):
            # create new path and copy
            file_name = file_prefix + str(img_id + 1) + "."\
                        + ImageFileUtil.get_file_extension(img_list[img_id]).lower()
            newimg_path = os.path.join(dest_path, file_name)
            shutil.copyfile(img_list[img_id], newimg_path)


def typename(o):

    module = ''
    class_name = ''
    if hasattr(o, '__module__') and o.__module__ != 'builtins' \
            and o.__module__ != '__builtin__' and o.__module__ is not None:
        module = o.__module__ + '.'

    if hasattr(o, '__qualname__'):
        class_name = o.__qualname__
    elif hasattr(o, '__name__'):
        class_name = o.__name__
    else:
        class_name = o.__class__.__name__

    return module + class_name
