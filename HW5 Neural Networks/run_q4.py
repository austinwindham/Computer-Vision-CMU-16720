import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)
    #print(bboxes)

    plt.imshow(bw, cmap = 'gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################
    rows = []
    rows.append([])
    # sort by bottom
    bboxes.sort(key=lambda x: x[2])
    bottom = bboxes[0][2]
    line = 0

    for box in bboxes:
        
        # check if need new line
        if(box[0] >= bottom):

            #make new bottom and line
            bottom = box[2]
            line +=1
            rows.append([])
            
        rows[line].append(box)



    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################
    ### needs to be after getting params since we are printing each line
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    ##########################
    ##### your code here #####
    ##########################

    for row in rows:
        
        # sort by x coord
        row.sort(key=lambda x:x[1])
        printline = ""
        
        for box in row:

            # box character
            minr, minc, maxr, maxc = box
            character = bw[minr:maxr, minc:maxc]

            # transpose, pad, resize, flattenlike above comment
            character = character.T
            character = np.pad(character, (20, 20), 'constant',constant_values=(1, 1))
            character = skimage.transform.resize(character,(32, 32))
            
            #print(character.shape)
            character = character.reshape(1, -1)

            ### Do NN
            h = forward(character, params, 'layer1')
            prob_dis = forward(h, params,'output',softmax)

            # find most likely and add to line
            loc = np.argmax(prob_dis[0,:])
            printline += letters[loc]

        print(printline)

    print('\n')
    