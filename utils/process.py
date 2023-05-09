from utils.landsat_downloader import download
from train.prediction import predict
import numpy as np
import os
from PIL import Image


DEFAULT_MODEL = "model-resnet50-novograd-0008.h5"
MIN_LONGITUDE_DELTA = 3.2
MIN_LATITUDE_DELTA = 1.6
DEPTH = 1


def date_to_interval(date):
    return "{0}T00:00:00Z".format(date), "{0}T23:59:59Z".format(date)


def get_mask(bbox, time_interval, model=DEFAULT_MODEL):
    assert (bbox[2] > bbox[0] and bbox[3] > bbox[1])
    """
    date is a string in format "yyyy-mm-dd"
    where y - year, m - month, d - day
    """
    if bbox[2] - bbox[0] > MIN_LONGITUDE_DELTA:
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
    elif bbox[3] - bbox[1] > MIN_LATITUDE_DELTA:
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
        x = download(bbox, time_interval, width=256*pow(2, DEPTH+1), height=256*pow(2, DEPTH+1), rescale=True)
        y = get_valid_area_mask(DEPTH, x, rescale=True, model=model)
        return y


def get_image(bbox, time_interval):
    assert (bbox[2] > bbox[0] and bbox[3] > bbox[1])
    """
    date is a string in format "yyyy-mm-dd"
    where y - year, m - month, d - day
    """
    if bbox[2] - bbox[0] > MIN_LONGITUDE_DELTA:
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
    elif bbox[3] - bbox[1] > MIN_LATITUDE_DELTA:
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
        x = get_valid_area_image(DEPTH, bbox, time_interval)
        return x


def get_valid_area_image(depth, bbox, time_interval):
    x = download(bbox, time_interval, width=256*depth, height=256*depth)
    x = x.astype(np.uint8)
    return x


def split_image(img):
    # splits image into 4 pieces

    xs = img.shape[0] // 2  # division lines for the picture
    ys = img.shape[1] // 2
    # now slice up the image (in a shape that works well with subplots)
    splits = [img[0:xs, 0:ys], img[0:xs, ys:], img[xs:, 0:ys], img[xs:, ys:]]
    return splits


def join_images(ul, ur, dl, dr):
    u = np.concatenate((ul, ur), axis=1)
    d = np.concatenate((dl, dr), axis=1)
    image = np.concatenate((u, d), axis=0)
    return image


def get_valid_area_mask(depth, x, rescale=True, model=DEFAULT_MODEL):
    batch = np.array(split_image(x))
    batch2 = list()
    for x in batch:
        split = split_image(x)
        for y in split:
            batch2.append(y)

    masks2 = predict(np.array(batch2), model)
    masks = list()
    for i in range(4):
        x = join_images(masks2[i*4], masks2[i*4+1], masks2[i*4+2], masks2[i*4+3])
        masks.append(x)
    masks = np.array(masks)
    mask = join_images(masks[0], masks[1], masks[2], masks[3])
    return mask

    """
    if depth > 0:
        masks = list()
        for img in batch:
            masks.append(get_valid_area_mask(depth-1, img, rescale, model))
        mask = join_images(masks[0], masks[1], masks[2], masks[3])
        return mask
    else:
        masks = predict(batch, model)
        mask = join_images(masks[0], masks[1], masks[2], masks[3])
        return mask
"""

