import cv2
import argparse

from cvtools import resize
from cvtools import perspective_transform
from cvtools import getoutlines
from cvtools import simple_erode

# Parse command-line arguments with argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image to be corrected.")
ap.add_argument("-I", "--inverted", required = False, nargs='?', const="Ture", help = "Invert the output if this argument present.")
args = vars(ap.parse_args())

# READ INPUT IMAGE
img = cv2.imread(args["image"])

# if input image is empty, notify the user and quit program
if img is None:
    print()
    print('The file does not exist or is empty!')
    print('Please select a valid file!')
    print()
    exit(0)  # exit code zero means a clean exit with no output/errors etc.


def preprocess(img):
    """
    BAISC PRE-PROCESSING TO OBTAIN A CANNY EDGE IMAGE
    """

    # save the original image in a different variable
    img_orig = img.copy()

    # first calculate the ratio of the original image to the new height (500)
    # so we can scale the manipulated image back to the original size;
    scale = img.shape[0] / 500.0

    # - scale the image down to 500px in height;
    img_scaled = resize(img, height = 500)

    img_gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)  # convert image to grayscale
    img_gray = cv2.GaussianBlur(img_gray, (7, 7), 0)         # apply gaussian blur
    img_edge = cv2.Canny(img_gray, 53, 200)                  # apply canny edge detection

    return img_orig, scale, img_scaled, img_edge


def gethull(img_edge):
    """
    1st ROUND OF OUTLINE FINDING, + CONVEX HULL
    """

    # make a copy of the edge image because the following function manipulates
    # the input
    img_hull = img_edge.copy()

    # find outlines in the edge image
    outlines = getoutlines(img_edge)

    # draw convex hulls (fit polygon) for all outlines detected to 'img_contour'
    for outline in range(len(outlines)):
        hull = cv2.convexHull(outlines[outline])
        cv2.drawContours(img_hull, [hull], 0, 255, 3)

    # erode the hull image to make the outline closer to paper
    img_hull = simple_erode(img_hull)

    return img_hull


def getcorners(img_hull):
    """
    2nd ROUND OF OUTLINE FINDING, + SORTING & APPROXIMATION
    """

    # make a copy of the edge image because the following function manipulates
    # the input
    img_outlines = img_hull.copy()

    # find outlines in the convex hull image
    outlines = getoutlines(img_outlines)

    # sort the outlines by area from large to small, and only take the largest 4
    # outlines in order to speed up the process and not waste time
    outlines = sorted(outlines, key = cv2.contourArea, reverse = True)[:4]

    # loop over outlines
    for outline in outlines:

        # find the perimeter of each outline for use in approximation
        perimeter = cv2.arcLength(outline, True)

        # > approximate a rough contour for each outline found, with (hopefully)
        #   4 points (rectangular sheet of paper); [Douglas-Peuker Algorithm]
        # > FIRST OPTION is the input outline;
        # > SECOND OPTION is the accuracy of approximation (epsilon), here it
        #   is set to a percentage of the perimeter of the outline
        # > THIRD OPTION is whether to assume an outline
        #   is closed, which in this case is yes (sheet of paper)
        approx = cv2.approxPolyDP(outline, 0.02 * perimeter, True)

        # if the approximation has 4 points, then assume it is correct, and
        # assign these points to the 'corners' variable
        if len(approx) == 4:
            corners = approx
            break

    return corners


'''
Main Proccess of the Program
'''

# obtain the Canny edge image, scaled, along with its scale factor
img_orig, scale, img_scaled, img_edge = preprocess(img)

# perform convex hull on edge image to prevent imcomplete outline
img_hull = gethull(img_edge)

# obtain 4 corner points of the convex hull image
corners = getcorners(img_hull)

# scale the corner points back to the original size of the image using the scale
# calculated previously
corners = corners.reshape(4, 2) * scale

# finally correct the perspective of the image by applying four-point
# perspective transform
img_corrected = perspective_transform(img_orig, corners)

# convert the corrected image to grayscale
img_corrected = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2GRAY)

# write corrected image to file
cv2.imwrite('./corrected.png', img_corrected)

# - conduct simple binary thresholding for ease of further processing; the first
#   option of the function is the actual threshold value
# - if given the argument "-I" or "--inverted", output an inverted binary image,
#   otherwise output a normal image
if args["inverted"] is not None:

    #img_thresh = cv2.threshold(img_corrected, 1, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    img_thresh = cv2.threshold(img_corrected, 150, 255, cv2.THRESH_BINARY_INV)[1]

    # write binary image to file
    cv2.imwrite('./thresholded_inverted.png', img_thresh)

else:

    #img_thresh = cv2.threshold(img_corrected, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    img_thresh = cv2.threshold(img_corrected, 135, 255, cv2.THRESH_BINARY)[1]

    # write binary image to file
    cv2.imwrite('./thresholded.png', img_thresh)
