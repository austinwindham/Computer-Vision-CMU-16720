'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''

import submission as sub
import helper
import util
import numpy as np
import matplotlib.pyplot as plt


# load data
data = np.load('../data/some_corresp.npz')

im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

pts1 = data['pts1']
pts2 = data['pts2']

intrinsic = np.load('../data/intrinsics.npz')
K1 = intrinsic['K1']
K2 = intrinsic['K2']

# get needed matrices

M = max(im1.shape)
F = sub.eightpoint(pts1,pts2,M)
E = sub.essentialMatrix(F,K1,K2)

M2s = helper.camera2(E)
# print(M2s.shape)
# print(M2s)

M1 = np.array([[1, 0, 0, 0], 
               [0, 1, 0, 0], 
               [0, 0, 1, 0]])

C1 = np.dot(K1, M1)

# for loop thorugh 4 solutions
current_error = np.inf
M2 = np.zeros((3,4))
C2_correct = np.zeros((3,4))
p = np.array(3)

for i in range(4):
    print(i)
    
    C2 = np.dot(K2, M2s[:, :, i])
    

    w, error = sub.triangulate(C1, pts1, C2,pts2)
    

    if error<current_error:
        M2 = M2s[:,:,i]
        C2_correct = C2
        P = w
        # print(M2)
        # print(error)
        print(i)
        print('last')
        current_error = error

# did P instead of w this time since report says P
np.savez('q3_3.npz', M2 =M2, C2 =C2_correct, P= P)

    

