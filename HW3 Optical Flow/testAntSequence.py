import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import SubtractDominantMotion
from SubtractDominantMotion import SubtractDominantMotion
import time
import scipy

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.05, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/antseq.npy')

frameshape = seq.shape
start = time.time()

for frame in range(frameshape[2]-1):
    print(frame)
    im2 = seq[:,:,frame+1]
    mask = SubtractDominantMotion(seq[:,:,frame], im2, threshold, num_iters, tolerance)
    mask = scipy.ndimage.morphology.binary_erosion(mask, iterations= 2)
    mask = scipy.ndimage.morphology.binary_dilation(mask, iterations=3)
    #mask = scipy.ndimage.morphology.binary_erosion(mask)
    # apply mask
    current = np.zeros((frameshape[0], frameshape[1],3))
    current[:,:,0] = im2
    current[:,:,0][mask ==1] =1
    current[:,:,1] = im2
    current[:,:,2] = im2

    cv2.imshow('image', current)
    # save as ICA or LKA
    if frame in [30,60,90,120]:
        cv2.imwrite('ant'+"ICA"+str(frame)+".png", current*255)
    cv2.waitKey(1)

end = time.time()

print("Elapsed Time:")
print(end-start)
