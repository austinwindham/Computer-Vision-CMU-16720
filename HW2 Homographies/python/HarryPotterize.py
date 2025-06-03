import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts

#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH

#Write script for Q2.2.4
opts = get_opts()


# read three images
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

# compute H using matchpics and ransac
matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)


#print(locs1.shape)
x1 = locs1[matches[:,0], 0:2]
x2 = locs2[matches[:,1],0:2]

x1 = x1[:,[1,0]]
x2 = x2[:,[1,0]]
#print(len(x1))
#print(len(x2))
#H2to1 = cv2.findHomography(x1,x2, cv2.RANSAC)[0]

#print(H2to1)
H2to1, inliers = computeH_ransac(x1,x2,opts)
print('homography matrix')
print(H2to1)
#print(inliers)
# warp hp_cover with opencv to cv_desk
## does it need to be inverse H? ask ta
height, width = cv_desk.shape[:2]
#warped_hp = cv2.warpPerspective(hp_cover, H2to1, (width, height))

# how would i modify  hp.cover.jpg, reshape, resize
hp_cover = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))
# implement compositeH
composite_img = compositeH(H2to1, hp_cover, cv_desk)

# show
cv2.imshow('composite',composite_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('harrypotter.jpg', composite_img)