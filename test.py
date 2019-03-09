import cv2

from cvtools import resize
from cvtools import perspective_transform
from cvtools import getoutlines
from cvtools import simple_erode
from cvtools import simple_dilate
from cvtools import brightness_contrast
from cvtools import blank

import matplotlib.pyplot as plt


img = cv2.imread('test/b.jpg')

# increase contrast between paper and background
img_adj = brightness_contrast(img, 1.56, -60)
# img_adj = cv2.convertScaleAbs(img, alpha=1.56, beta=-60)


def pltimg(img):
    plt.figure(figsize=(15,15))
    plt.axis("off")
    plt.imshow(img, cmap='gray')
    plt.show()


def pltcolor(img):
    plt.figure(figsize=(15,15))
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


pltcolor(img_adj)
