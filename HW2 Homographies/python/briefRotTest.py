import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
import scipy
import matplotlib.pyplot as plt
from helper import plotMatches

opts = get_opts()
#Q2.1.6
#Read the image and convert to grayscale, if necessary
cover = cv2.imread('../data/cv_cover.jpg')

degrees = []
matches_per_degree = []

for i in range(36):
	#Rotate Image
	deg = i*10
	print(deg)
	cover_rotated = scipy.ndimage.rotate(cover, deg)

	#Compute features, descriptors and Match features
	matches, locs1, locs2 = matchPics(cover, cover_rotated, opts)
	
	
	#Update histogram
	
	degrees.append(deg)
	matches_per_degree.append(len(matches))

	


#Display histogram

plt.bar(degrees, matches_per_degree, width=8, align='center', color='blue')


plt.xlabel('Rotation Angle')
plt.ylabel('Number of Matches')
plt.title('Matches per Rotation Angle')


plt.show()