import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]


### Same process, but provide steps from paper
### Need pn, pnstar, keep first frame around, apply if statement
### keep ttrack of orignial and updated template and pnot when needed
### Print with both rectangles later

total_frames = seq.shape[2]

car_rects = rect
orignial_rect = rect[:]

original_image = seq[:,:,0]
template_image = seq[:,:,0]
p0 = np.zeros(2)
### Test out epsilon
epsilon = template_threshold

for frame in range(total_frames-1):
    print(frame)

    # get current image and rect shift aleady
    current_image = seq[:,:, frame+1]
    rect_moved = [rect[0]-orignial_rect[0], rect[1]-orignial_rect[1]]

    # normal p
    p = LucasKanade(template_image, current_image, rect, threshold, num_iters, p0=p0)

    # pn 
    pn = [p[0]+rect_moved[0], p[1]+rect_moved[1]] 

    # pnstar
    pnstar = LucasKanade(original_image, current_image, orignial_rect, threshold, num_iters, p0=pn)

    # Find norm of pnstar and pn
    error = np.linalg.norm(pnstar - pn)

    # if statement section
    if error<= epsilon:
        # update template, set pnot to 0
        p0 = np.zeros(2)
        template_image = seq[:,:,frame+1]
        rect[0] = orignial_rect[0]+pnstar[0]
        rect[1] = orignial_rect[1] + pnstar[1]
        rect[2] = orignial_rect[2]+pnstar[0]
        rect[3] = orignial_rect[3] + pnstar[1]
        car_rects = np.vstack((car_rects, rect))


    else:
        # big error, don't update template and set pnot to p, rect must stay the same since template doesn't change
        p0 = p
        rx1 = rect[0] + p[0]
        rx2 = rect[1] + p[1]
        rx3 = rect[2] + p[0]
        rx4 = rect[3] + p[1]
        rects_notupdated = [rx1,rx2,rx3,rx4]

        car_rects = np.vstack((car_rects, rects_notupdated))





np.save('carseqrects-wcrt.npy', car_rects)

# plotter
seq = np.load("../data/carseq.npy")
car = np.load('carseqrects.npy')
car_corrected = np.load('carseqrects-wcrt.npy')

for i in range(len(car)):
    if i in [0, 99, 199, 299, 399]:
        print(car[i])
        print(car_corrected[i])
        carrect = car[i]
        carrect2 = car_corrected[i]
        

        fig = plt.figure()
        plt.imshow(seq[:,:,i], cmap='gray')
        plt.axis('off')
        plt.axis('tight')
        #print(rect)
        patch = patches.Rectangle((carrect[0], carrect[1]), (carrect[2]-carrect[0]),
                                  (carrect[3]-carrect[1]), edgecolor='b', facecolor='none', linewidth=1)
        
        patch2 = patches.Rectangle((carrect2[0], carrect2[1]), (carrect2[2]-carrect2[0]),
                                  (carrect2[3]-carrect2[1]), edgecolor='r', facecolor='none', linewidth=1)
        ax = plt.gca()
        ax.add_patch(patch)
        ax.add_patch(patch2)
        fig.savefig('carcorrected'+str(i+1)+'.png', bbox_inches='tight')


