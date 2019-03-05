import cv2
import argparse

from cvtools import resize
from cvtools import perspective_transform

# Parse command-line arguments with argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image to be corrected.")
ap.add_argument("-I", "--inverted", required = False, nargs='?', const="Ture", help = "Invert the output if this argument present.")
args = vars(ap.parse_args())

# READ INPUT IMAGE
# img = cv2.imread('notecard.png')
img = cv2.imread(args["image"])

# BAISC PRE-PROCESSING TO OBTAIN A CANNY EDGE IMAGE
def getedge(img):
    # save the original image in a different variable
    img_orig = img

    # first calculate the ratio of the original image to the new height (500)
    # so we can scale the manipulated image back to the original size;
    scale = img.shape[0] / 500.0

    # - scale the image down to 500px in height;
    img_scaled = resize(img, height = 500)

    img_gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)  # convert image to grayscale
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)         # apply gaussian blur
    img_edge = cv2.Canny(img_gray, 60, 200)                 # apply canny edge detection

    return img_orig, scale, img_scaled, img_edge


# FUNCTION TO OBTAIN THE CORNERS OF A PIECE OF PAPER IN THE INPUT EDGE IMAGE
def getcorners(img_edge):
    # make a copy of the edge image because the following function manipulates
    # the input
    img_contour = img_edge

    # - find the possible oulines of the edge image using a built-in OpenCV function;
    # - the "RETR_LIST" retreival mode returns a simple list of the found
    #   outlines;
    # - the "cv2.CHAIN_APPROX_SIMPLE" approximation method returns coordinate points
    #   for the found outlines;
    # - because the return of the contour function gives 'image', 'contours',
    #   'heirarchy', we will only take the contours (outlines) for the current
    #   application
    img_contour, outlines, contour_heirarchy = cv2.findContours(img_contour, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

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

# Obtain the Canny edge image, scaled, along with its scale factor
img_orig, scale, img_scaled, img_edge = getedge(img)

# Obtain corner points of the scaled image
corners = getcorners(img_edge)

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
if args["inverted"] != None:

    #img_thresh = cv2.threshold(img_corrected, 1, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    img_thresh = cv2.threshold(img_corrected, 150, 255, cv2.THRESH_BINARY_INV)[1]

    # write binary image to file
    cv2.imwrite('./thresholded_inverted.png', img_thresh)

else:

    #img_thresh = cv2.threshold(img_corrected, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    img_thresh = cv2.threshold(img_corrected, 135, 255, cv2.THRESH_BINARY)[1]

    # write binary image to file
    cv2.imwrite('./thresholded.png', img_thresh)

