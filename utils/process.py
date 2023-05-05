from utils.landsat_downloader import download
from train.prediction import predict
import numpy as np
import os
from PIL import Image


DEFAULT_MODEL = "model-resnet50-novograd-0008.h5"


def date_to_interval(date):
    return "{0}T00:00:00Z".format(date), "{0}T23:59:59Z".format(date)


def get_mask(bbox, time_interval, model=DEFAULT_MODEL):
    assert (bbox[2] > bbox[0] and bbox[3] > bbox[1])
    """
    date is a string in format "yyyy-mm-dd"
    where y - year, m - month, d - day
    """
    if bbox[2] - bbox[0] > 0.8:
        im1 = get_mask(
            [bbox[0],
             bbox[1],
             (bbox[0] + bbox[2]) / 2,
             bbox[3],
             ],
            time_interval)
        im2 = get_mask(
            [(bbox[0] + bbox[2]) / 2,
             bbox[1],
             bbox[2],
             bbox[3],
             ],
            time_interval)
        return np.concatenate((im1, im2), axis=1)
    elif bbox[3] - bbox[1] > 0.4:
        im1 = get_mask(
            [bbox[0],
             bbox[1],
             bbox[2],
             (bbox[3] + bbox[1])/2,
             ],
            time_interval)
        im2 = get_mask(
            [bbox[0],
             (bbox[3] + bbox[1])/2,
             bbox[2],
             bbox[3],
             ],
            time_interval)
        return np.concatenate((im2, im1), axis=0)
    else:
        x = download(bbox, time_interval, True)
        batch = np.array([x, ])
        y = predict(batch, model)[0]
        return y


def get_image(bbox, time_interval):
    assert (bbox[2] > bbox[0] and bbox[3] > bbox[1])
    """
    date is a string in format "yyyy-mm-dd"
    where y - year, m - month, d - day
    """
    if bbox[2] - bbox[0] > 0.8:
        im1 = get_image(
            [bbox[0],
             bbox[1],
             float((bbox[0] + bbox[2]) / 2),
             bbox[3],
             ],
            time_interval)
        im2 = get_image(
            [float((bbox[0] + bbox[2]) / 2),
             bbox[1],
             bbox[2],
             bbox[3],
             ],
            time_interval)
        return np.concatenate((im1, im2), axis=1)
    elif bbox[3] - bbox[1] > 0.4:
        im1 = get_image(
            [bbox[0],
             bbox[1],
             bbox[2],
             float((bbox[3] + bbox[1]) / 2),
             ],
            time_interval)
        im2 = get_image(
            [bbox[0],
             float((bbox[3] + bbox[1]) / 2),
             bbox[2],
             bbox[3],
             ],
            time_interval)
        return np.concatenate((im2, im1), axis=0)
    else:
        x = download(bbox, time_interval, False)
        x = x.astype(np.uint8)
        return x

