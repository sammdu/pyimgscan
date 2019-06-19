import numpy as np
import cv2

"""
List of functions:

* blank(shape, dtype = np.uint8, filler = '0')
    - creates a blank image (NumPy array) with either '0's or '1's

* simple_erode(img)
    - apply erosion with a 3x3 kernel and for 1 single iteration

* simple_dilate(img)
    - apply dilation with a 3x3 kernel and for 1 single iteration

* brightness_contrast(img, mult, add)
    - adjust the brightness and contrast of the image by multiplying and
      adding/subtracting values from the pixels;

* resize(img, width=None, height=None, inter=cv2.INTER_AREA)
    - resizes the image to the input height, width, or both

* getoutlines(img)
    - retrieve outlines (contours) of an input image

* order_points(pts)
    - arrange a list of corner points by coordinates, by the order of
      TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT

* perspective_transform(img, pts)
    - warps a deformed image into a straight, birds-eye view image,
      based on the corner points on the deformed image
"""


def blank(shape, dtype=np.uint8, filler="0"):
    """
    - creates a blank image (NumPy array) with either zeros or ones using
      built-in NumPy functions;
    - "shape" sets the shape of the blank image (array); takes tuples;
    - "dtype" sets the data type of the blank image, defaults to 8-bit
      unsigned integer; takes NumPy data-type objects;
    - "filler" sets the filler value of all pixels of the blank image, defaults
      to zeros; takes strings of either "0" or "1"
    """

    if filler == "0":
        blank = np.zeros(shape, dtype)

    elif filler == "1":
        blank = np.ones(shape, dtype)

    else:
        return "BAD FILLER VALUE; MUST BE STRINGS OF '0' OR '1'"

    return blank


def simple_erode(img):
    """
    - apply simple erosion to the input image using built-in OpenCV functions;
    - erosion kernel 3x3;
    - 1 single iteration
    """

    ekernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(img, ekernel, iterations=1)

    return eroded


def simple_dilate(img):
    """
    - apply simple dilation to the input image using built-in OpenCV functions;
    - dilation kernel 3x3;
    - 1 single iteration
    """

    dkernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(img, dkernel, iterations=1)

    return dilated


def brightness_contrast(img, mult, add):
    """
    - allows brightness-contrast adjustment by multiplication and
      addition / subtraction of pixel values;
    - multiplication increases contrast while inevitably increasing overall
      brightness; set by "mult" parameter;
    - addition or subtraction increases or decreases value (brightness) of
      pixels; set by "add" parameter, use negative values for subtraction;
    """

    # multiply pixels by "mult", add by "add";
    # the multiplication will increase contrast, and adding/subtracting will
    # adjust the brightness
    adjusted = cv2.convertScaleAbs(img, alpha=float(mult), beta=float(add))

    return adjusted


# imutils resize() function.

"""
https://github.com/jrosebr1/imutils
MIT License.

Copyright (c) 2015-2016 Adrian Rosebrock, http://www.pyimagesearch.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


def resize(img, width=None, height=None, inter=cv2.INTER_AREA):
    """
    initialize the dimensions of the input image and obtain
    the image size
    """

    dim = None
    (h, w) = img.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return img

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(img, dim, interpolation=inter)

    # return the resized image
    return resized


def getoutlines(img):
    """
    - find the possible oulines of the input image using a built-in OpenCV
      function;
    - the "RETR_LIST" retreival mode returns a simple list of the found
      outlines;
    - the "cv2.CHAIN_APPROX_SIMPLE" approximation method returns coordinate
      points for the found outlines;
    - because the return of the contour function gives "contours", "heirarchy",
      we will only take the contours (outlines) for the current application
    """

    outlines = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    return outlines


# 4-POINT PERSPECTIVE TRANSFORM FUNCTION SET.

"""
 - Call function "perspective_transform()";
 - Supply an input image, then 4 corner points;
 - The function will return the corrected image;
 - order_points() is a helper function
"""


def order_points(pts):
    """
    - returns a list of corner points in order
    - input "pts" will be a numpy array; will convert python list automatically;
    - "corners" list will go in the following order:
      0.    TOP-LEFT
      1.    TOP-RIGHT
      2. BOTTOM-RIGHT
      4. BOTTOM-LEFT
    """

    # automatically convert a python list to a numpy array
    if str(type(pts)) != "<class 'numpy.ndarray'>":
        pts = np.array(pts)

    # initialize an empty list to store the corner points
    corners = np.zeros((4, 2), dtype="float32")

    # figure out which set of coordinates are at which corner;

    # > the TOP-LEFT coordinates will have the smallest sum
    # > the BOTTOM-RIGHT coordinates will have the largest sum
    sums = pts.sum(axis=1)  # sum up all numbers horizontally
    corners[0] = pts[np.argmin(sums)]  # find out the TOP-LEFT coordinate
    corners[2] = pts[np.argmax(sums)]  # find out the BOTTOM-RIGHT coordinate

    # > the TOP-RIGHT coordinates will have the smallest difference
    # > the BOTTOM-LEFT coordinates will have the largest difference
    diffs = np.diff(pts, axis=1)
    corners[1] = pts[np.argmin(diffs)]
    corners[3] = pts[np.argmax(diffs)]

    return corners


def perspective_transform(img, pts):
    """
    - applies perspective transform to an image in order to straighten it,
      based on four given corner points
    - input "pts" will be a numpy array
    - returns a corrected image after applying perspective transform
    """

    # call the "order_points()" function to put corner points in order,
    # then assign each corner point to its respective variable
    corners_old = order_points(pts)
    tl, tr, br, bl = corners_old

    # calculate the WIDTH of the corrected image as follows:
    # > find the distance between the top points and the bottom points;
    # > find the maximum of the two distances, make it the new width.
    distT = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    distB = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    maxW = max(int(distT), int(distB))

    # calculate the HEIGHT of the corrected image as follows:
    # > find the distance between the left points and the right points;
    # > find the maximum of the two distances, make it the new height.
    distL = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    distR = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    maxH = max(int(distL), int(distR))

    # define corner points for the corrected image based on calculations
    # done above;
    # the same order of corners are followed
    corners_corrected = np.array(
        [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32"
    )

    # using OpenCV, calculate a transform matrix then apply it to create
    # the corrected image
    matrix = cv2.getPerspectiveTransform(corners_old, corners_corrected)
    img_corrected = cv2.warpPerspective(img, matrix, (maxW, maxH))

    return img_corrected
