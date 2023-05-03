from utils.landsat_downloader import download
from utils import show
from train.prediction import predict
import numpy as np
import os
from utils.process import get_mask, get_image


def print_hi(name):
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

    bbox = [
      45.344712,
      55.764213,
      47.384364,
      56.824933
    ]
    """
    bbox = [45.333566,
            56.139429,
            45.947059,
            56.501923
            ]
            """
    time_interval = ("2022-08-15T00:00:00Z", "2022-08-30T00:00:00Z")
    """
    x = get_image(bbox, time_interval)
    show.show(show.rgb_410(x))
    """
    y = get_mask(bbox, time_interval)
    print(y.shape)
    show.show(y)



if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
