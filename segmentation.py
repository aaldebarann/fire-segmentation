import utils.show
from utils.process import get_mask, get_image
import matplotlib.pyplot as plt
import numpy as np


def main():
    bbox = [
      45.526559,
      56.323391,
      45.722638,
      56.418649
    ]
    time_interval = ("2022-08-15T00:00:00Z", "2022-08-30T00:00:00Z")

    x = get_image(bbox, time_interval)  # 5-channels image
    x = utils.show.rgb_410(x)  # RGB image
    y = get_mask(bbox, time_interval)

    fig = plt.figure()
    plt.axis('off')
    plt.title("fire segmentation sample")
    a = fig.add_subplot(1, 2, 1)
    plt.imshow(x, interpolation='lanczos')
    a.set_title('image')
    a = fig.add_subplot(1, 2, 2)
    plt.imshow(y, interpolation='nearest')
    a.set_title('mask')
    plt.show()


if __name__ == '__main__':
    main()
