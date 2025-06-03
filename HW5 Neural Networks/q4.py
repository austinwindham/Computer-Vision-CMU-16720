import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation
import scipy.ndimage
import matplotlib.pyplot as plt

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    # deonise
    image = scipy.ndimage.gaussian_filter(image, sigma=(2, 2, 0), mode='constant')

    # grayscale
    image= skimage.color.rgb2gray(image)

    # threshold
    threshold = skimage.filters.threshold_otsu(image)

    # morphology
    bw = skimage.morphology.closing(image <threshold, skimage.morphology.square(8))
    clearborder= skimage.segmentation.clear_border(bw)

    bw = ~clearborder
    

    # label
    labels, num = skimage.measure.label(clearborder, background=0, return_num=True, connectivity=2 )

    # skip small boxes
    pad = 0
    for i in skimage.measure.regionprops(labels):
        if i.area >= 200:
            minr, minc, maxr, maxc = i.bbox
            minr = max(0, minr - pad)
            minc = max(0, minc - pad)
            maxr = min(bw.shape[0], maxr + pad)
            maxc = min(bw.shape[1], maxc + pad)
            bboxes.append((minr, minc, maxr, maxc))

    # plt.imshow(bw, cmap='gray', origin='upper')
    # plt.show()
    
    return bboxes, bw