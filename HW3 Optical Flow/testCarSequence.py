import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]


# loop through frames calling LK

total_frames = seq.shape[2]
print(seq.shape)
car_rects = rect

for frame in range(total_frames-1):
    # get template image and current image
    print(frame)
    template_image = seq[:,:, frame]
    current_image = seq[:,:, frame+1]

    # call Lk, update rectange and save
    p = LucasKanade(template_image, current_image, rect, threshold, num_iters)
    #print(p)
    rect[0] += p[0]
    rect[1] += p[1]
    rect[2] += p[0]
    rect[3] += p[1]

    car_rects = np.vstack((car_rects, rect))

    # Save frames 1, 100, 200, 300, 400 with drawn rectangle
    
    if frame in [0, 99, 199, 299, 399]:
        #print(p)
        fig = plt.figure()
        plt.imshow(current_image, cmap='gray')

        # remove grid and whitespace
        plt.axis('tight')
        plt.axis('off')

        print(rect)
        patch = patches.Rectangle((rect[0], rect[1]), (rect[2]-rect[0]),(rect[3]-rect[1]), edgecolor='r', facecolor='none', linewidth=2)
        ax = plt.gca()
        ax.add_patch(patch)
        fig.savefig('car'+str(frame+1)+'.png', bbox_inches='tight')


np.save('carseqrects.npy', car_rects)
