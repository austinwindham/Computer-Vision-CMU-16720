import numpy as np
import cv2

from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH

# Need to harrypotterize, but with right image as the desk and need to make blank space on left for left image to combine

from opts import get_opts
opts = get_opts()

# load left and right
left = cv2.imread('IMG_3824.jpg')
right = cv2.imread('IMG_3825.jpg')

# image two big dived all dimensions by 4

left_img = cv2.resize(left, (left.shape[1]//4, left.shape[0]//4))
right_img= cv2.resize(right, (right.shape[1]//4, right.shape[0]//4))

# get shape paramaters
LH, LW = left_img.shape[0:2]
RH, RW = right_img.shape[0:2]

print(left_img.shape)
print(right_img.shape)
W = round(LW*1.5)

# make new right with border on left that is 1.5 size, heihgts shold be same anyway
right_new = cv2.copyMakeBorder(right_img, 0, RH-LH, W-RW, 0, cv2.BORDER_CONSTANT, 0)

# test right side
# cv2.imshow('right_pad', right_new)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# hp stuff
matches, locs1, locs2 = matchPics(left_img, right_new, opts)

x1 = locs1[matches[:,0], 0:2]
x2 = locs2[matches[:,1],0:2]

x1 = x1[:,[1,0]]
x2 = x2[:,[1,0]]
print('before H')
bestH2to1, inliers = computeH_ransac(x1, x2, opts)
print('after h')
panorama = compositeH(bestH2to1, left_img, right_new)

# get best from two images
panorama = np.maximum(right_new, panorama)

# cv2.imshow('panorama', panorama)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite('panorama.png', panorama)