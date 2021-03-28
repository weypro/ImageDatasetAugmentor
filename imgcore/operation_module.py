from abc import abstractmethod, ABCMeta
import random
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
import numpy as np
from numpy.core.multiarray import ndarray
import cv2
import operator
from collections import OrderedDict
from itertools import islice

import time

from utils import utils


T = TypeVar('T', bound='OperationModule')


class OperationModule:
    r"""Base class for all operation modules.

    OperationModules can also contain other OperationModules, allowing to nest them in
    a tree structure. It is almost the same as "module" in "pytorch".
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
        self._modules = OrderedDict()

    @abstractmethod
    def execute(self, image_array: ndarray):
        ...

    def __call__(self, *args):
        return self.execute(*args)

    def add_module(self, name: str,
                   module: Optional['OperationModule']) -> None:
        """Adds a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (string): name of the child module. The child module can be
                accessed from this module using the given name
            module (OperationModule): child module to be added to the module.
        """
        if module is None:
            raise TypeError("{} is None".format(utils.typename(module)))
        elif not isinstance(module, OperationModule) and module is not None:
            raise TypeError("{} is not a Module subclass".format(
                utils.typename(module)))
        elif not isinstance(name, str):
            raise TypeError("module name should be a string. Got {}".format(
                utils.typename(name)))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError(
                "module name can't contain \".\", got: {}".format(name))
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")

        # Add successfully
        self._modules[name] = module


class Sequential(OperationModule):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def execute(self, input):
        for module in self:
            input = module(input)
        return input

    def _get_item_by_idx(self, iterator, idx) -> T:
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx) -> Union['Sequential', T]:
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(
                list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx: int, module: OperationModule) -> None:
        key: str = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: Union[slice, int]) -> None:
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self) -> int:
        return len(self._modules)

    def __dir__(self):
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def __iter__(self) -> Iterator[OperationModule]:
        return iter(self._modules.values())


class RandomRotate(OperationModule):
    max_left_degree = None
    max_right_degree = None

    def __init__(self, max_left_degree: int, max_right_degree: int) -> None:
        super().__init__()
        self.max_left_degree = max_left_degree
        self.max_right_degree = max_right_degree
        random_degree = random.uniform(-self.max_right_degree,
                                       self.max_left_degree)
        self.add_module("rotate1", Rotate(random_degree))

    def execute(self, image_array: ndarray):
        return self._modules["rotate1"].execute(image_array)


class Rotate(OperationModule):
    degree = None

    def __init__(self, degree: int) -> None:
        super().__init__()
        self.degree = degree

    def execute(self, image_array: ndarray):
        height, width = image_array.shape[:2]
        M = cv2.getRotationMatrix2D((width / 2, height / 2), self.degree, 1)
        return cv2.warpAffine(image_array, M, (width, height))


class SaltPepperNoise(OperationModule):
    proportion = None

    def __init__(self, proportion: int) -> None:
        super().__init__()
        self.proportion = proportion

    def execute(self, image_array: ndarray):
        height, width = image_array.shape[:2]
        # Calculate the count of noise
        num = int(height * width * self.proportion)

        # Add salt-and-pepper noise randomly in certain times
        for i in range(num):
            w = random.randint(0, width - 1)
            h = random.randint(0, height - 1)
            if random.randint(0, 1) == 0:
                image_array[h, w] = 0
            else:
                image_array[h, w] = 255
        return image_array


class GaussianNoise(OperationModule):
    mean = None
    sigma = None

    def __init__(self, mean: int, sigma: int) -> None:
        super().__init__()
        self.mean = mean
        self.sigma = sigma

    def execute(self, image_array: ndarray):
        def clamp(n, minn=0, maxn=255):
            # Clamp the value between an upper and lower bound
            return max(min(maxn, n), minn)

        # Add gaussian noise to every color element
        # The iterator is faster
        for element in np.nditer(image_array, op_flags=['readwrite']): 
            # Should limit the range before assigning, or it will overflow
            element[...] = clamp(
                element +
                np.random.normal(self.mean, self.sigma))

        return image_array


class Blur(OperationModule):
    def __init__(self) -> None:
        super().__init__()

    def execute(self, image_array: ndarray):
        return cv2.medianBlur(image_array, 5)


class Shift(OperationModule):
    width_offset = None
    height_offset = None

    def __init__(self, width_offset: int, height_offset: int) -> None:
        super().__init__()
        self.width_offset = width_offset
        self.height_offset = height_offset

    def execute(self, image_array: ndarray):
        M = np.float32([[1, 0, self.width_offset], [0, 1, self.height_offset]])
        return cv2.warpAffine(image_array, M,
                              (image_array.shape[1], image_array.shape[0]))


class Resize(OperationModule):
    def __init__(self, width: int, heigth: int) -> None:
        super().__init__()
        self.width = width
        self.heigth = heigth

    def execute(self, image_array: ndarray):
        return cv2.resize(image_array, (self.width, self.heigth))


class HorizontalFlip(OperationModule):
    def __init__(self) -> None:
        super().__init__()

    def execute(self, image_array: ndarray):
        return cv2.flip(image_array, 1)


class VerticalFlip(OperationModule):
    def __init__(self) -> None:
        super().__init__()

    def execute(self, image_array: ndarray):
        return cv2.flip(image_array, 0)



class PolygonClipping(OperationModule):
    points = None

    def __init__(self, points: List) -> None:
        super().__init__()
        self.points = points

    def execute(self, image_array: ndarray):
        points = np.array([self.points])
        # Create a mask array of the same size
        mask = np.zeros(image_array.shape[:2], np.uint8)
        # Create the region and fill it with white color
        cv2.polylines(mask, points, 1, 255)
        cv2.fillPoly(mask, points, 255)
        # Bitwise and. Get the new image with black background
        dst = cv2.bitwise_and(image_array, image_array, mask=mask)
        # Add white color
        # bg = np.ones_like(img, np.uint8) * 255
        # cv2.bitwise_not(bg, bg, mask=mask)
        # dst_white = bg + dst
        return dst


class RectClipping(OperationModule):
    x1 = None
    y1 = None
    x2 = None
    y2 = None

    def __init__(self, x1: int, y1: int, x2: int, y2: int) -> None:
        super().__init__()
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def execute(self, image_array: ndarray):
        return image_array[self.y1:self.y2, self.x1:self.x2]
        

class ImgSave(OperationModule):
    path = None
    file_prefix = None

    def __init__(self, path: str, file_prefix: Optional[str] = "") -> None:
        super().__init__()
        self.path = path
        self.file_prefix = file_prefix

    def execute(self, image_array: ndarray):
        utils.ImageFileUtil.save_file(image_array, self.path, self.file_prefix)
        return image_array
