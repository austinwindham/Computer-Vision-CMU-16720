'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''

import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper
import util


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


M1 = np.array([[1, 0, 0, 0], 
               [0, 1, 0, 0], 
               [0, 0, 1, 0]])

C1 = np.dot(K1, M1)

templevalues = np.load('../data/templeCoords.npz')

x1 = templevalues['x1'][:, 0]
y1 = templevalues['y1'][:, 0]



# get second pair of points


temple2 =[]
for i in range(x1.shape[0]):

    x2, y2 = sub.epipolarCorrespondence(im1, im2, F, x1[i], y1[i])

    temple2.append([x2, y2])

temple1 = np.column_stack((x1,y1))
temple2 = np.array(temple2)

# get 3d points

current_error = np.inf
M2 = np.zeros((3,4))
C2_correct = np.zeros((3,4))
p = np.array(3)

for i in range(4):
    
    C2 = np.dot(K2, M2s[:, :, i])
    

    w, error = sub.triangulate(C1, temple1, C2,temple2)
    

    if error<current_error:
        M2 = M2s[:,:,i]
        C2_correct = C2
        P = w
        current_error = error




np.savez('q4_2.npz', F=F, M1=M1, M2=M2, C1=C1, C2=C2_correct)

# plotting

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(P[:,0], P[:,1], P[:,2], c='b',marker='.')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()