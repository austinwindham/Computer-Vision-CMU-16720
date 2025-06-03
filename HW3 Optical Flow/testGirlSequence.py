import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
# default is 1e-2
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
    
seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]

# loop through frames calling LK

total_frames = seq.shape[2]
#print(seq.shape)
girl_rects = rect

for frame in range(total_frames-1):
    print(frame)
    # get template image and current image
    template_image = seq[:,:, frame]
    current_image = seq[:,:, frame+1]

    # call Lk, update rectange and save
    p = LucasKanade(template_image, current_image, rect, threshold, num_iters)
    #print(p)
    rect[0] += p[0]
    rect[1] += p[1]
    rect[2] += p[0]
    rect[3] += p[1]

    girl_rects = np.vstack((girl_rects, rect))

    # Save frames 1, 100, 200, 300, 400 with drawn rectangle
    
    if frame in [0, 19, 39, 59, 79]:
        print(frame)
        #print(p)
        fig = plt.figure()
        plt.imshow(current_image, cmap='gray')
        plt.axis('off')
        plt.axis('tight')
        #print(rect)
        patch = patches.Rectangle((rect[0], rect[1]), (rect[2]-rect[0]),(rect[3]-rect[1]), edgecolor='r', facecolor='none', linewidth=2)
        ax = plt.gca()
        ax.add_patch(patch)
        fig.savefig('girl'+str(frame+1)+'.png', bbox_inches='tight')


np.save('girlseqrects.npy', girl_rects)
