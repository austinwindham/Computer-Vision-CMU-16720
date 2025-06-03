"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import util
import helper
import matplotlib.pyplot as plt


'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):

    x1 = pts1[:,0]/M
    y1 = pts1[:,1]/M
    
    x2 = pts2[:,0]/M
    y2 = pts2[:,1]/M

    T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])

    # make A and then SVD
    
    A = np.vstack((x2*x1, x2*y1, x2, y2*x1,  y2*y1, y2, x1, y1, np.ones(pts1.shape[0]))).T

    U, S, V, = np.linalg.svd(A)

    # use utility functions and rescale

    F = V[-1].reshape(3,3)

    F = util.refineF(F, pts1/M, pts2/M)

    F = util._singularize(F)

    F = T.T @ F @ T


    return F

# test this

# data = np.load('../data/some_corresp.npz')
# pts1 = data['pts1']
# pts2 = data['pts2']
# im1 = plt.imread('../data/im1.png')
# im2 = plt.imread('../data/im2.png')
# M = np.max(im1.shape)

# F = eightpoint(pts1, pts2, M) 
# print(F)
# #helper.displayEpipolarF(im1, im2, F) 


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    
    # Find E

    E = K2.T @ F @ K1

    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
   # report says return w, above says return P, they are the sme thing, I went with w since I used it in my 
   # equations for the report 

    w = np.empty((pts1.shape[0], 3))
    err =0

    x1 = pts1[:,0]
    y1 = pts1[:,1]
    
    x2 = pts2[:,0]
    y2 = pts2[:,1]

    # loop to get A

    for i in range(pts1.shape[0]):
        
        
        A = np.vstack((x1[i]*C1[2,:] - C1[0,:],
                        y1[i]*C1[2,:] - C1[1,:],
                        x2[i]*C2[2,:] - C2[0,:], 
                        y2[i]*C2[2,:] - C2[1,:]))
        
        U, S, V = np.linalg.svd(A)

        w[i,:] = V[-1][0:3]/V[-1][-1]

    # add up errors

    whomog = np.hstack((w, np.ones((pts1.shape[0], 1))))
    
    for i in range(pts1.shape[0]):

        x1proj = C1 @ whomog[i,:].T
        x2proj =  C2 @ whomog[i,:].T
        # make 2 values
        x1proj = np.transpose(x1proj[:2]/abs(x1proj[-1]))
        x2proj = np.transpose(x2proj[:2]/abs(x2proj[-1]))
        
        err += np.sum((x1proj-pts1[i])**2 + (x2proj-pts2[i])**2)
        

    return w, err


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    
    # find epipolar line and the find im2 coords in subsection with gaussian weight

    # vary these two
    window = 20
    sigma = 5

    x1 = int(x1)
    y1 = int(y1)

    epipolar_line = F@ np.array([x1, y1, 1]).T
    

    window1 = im1[(y1-window//2): (y1+window//2 +1), (x1-window//2): (x1+window//2 +1), :]

    
    window_x, window_y = np.meshgrid(np.arange(-window//2, window//2 +1, 1), np.arange(-window//2, window//2 +1, 1))

    weight = np.exp((window_x**2 + window_y**2) /(-2*(sigma**2)))/np.sqrt(2*np.pi*(sigma**2))
    
    weight /= np.sum(weight)

    weight = np.dstack([weight, weight, weight])

    current_error = np.inf

    for y2 in range((y1-window//2), (y1+window//2 + 1)):
        # from ax+by+c = 0
        x2 = int((-epipolar_line[1]*y2 - epipolar_line[2]) / epipolar_line[0])
    
        
        window2 = im2[y2-window//2: y2+window//2 +1, x2-window//2 :x2+window//2 +1, :]

        error  = np.linalg.norm((window1 -window2)*weight)
        if error < current_error:
            x2_best = x2
            y2_best = y2
            current_error = error

    return x2_best, y2_best



'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=0.42):
    
    

    x1 = np.vstack((pts1.T, np.ones(pts1.shape[0])))
    

    x2 = np.vstack((pts2.T, np.ones(pts1.shape[0])))

    max =0

    for i in range(nIters):
        
        random_values = np.random.choice(x1.shape[0], 8)
       

        F = eightpoint(pts1[random_values,:], pts2[random_values,:], M)
        
        x2hat = F @x1
        x2hat = x2hat/np.sqrt(np.sum(x2hat[:2, :]**2, axis=0))

        
        error = abs(np.sum(x2*x2hat, axis=0))
        
        inliers_current = error < tol
        inlier_amount = np.sum(inliers_current)


        if inlier_amount> max:
            F_best = F
            max = inlier_amount
            inliers = inliers_current

    return F_best, inliers

'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):

    theta = np.linalg.norm(r)
    # need no rotation case
    if theta ==0:
        return np.eye(3) 

    k = r/theta 

    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])

    R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*np.dot(K,K)

    return R
'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    
    cos_theta = (np.trace(R)-1)/2
    # handles issues if cos(th) outsid eof bounds
    cos_theta = min(1,max(-1, cos_theta))  
    theta = np.arccos(cos_theta)

    #handle no rotation case
    if np.isclose(theta,0):
        return np.zeros((3, 1))  

    sin_theta = np.sin(theta)

    # find r vector
    rx =(1/(2*sin_theta) )*(R[2,1] - R[1,2])
    ry =(1/(2*sin_theta)) *(R[0,2] - R[2,0])
    rz = (1/(2*sin_theta)) *(R[1,0] - R[0,1])

    r = np.array([[rx],[ry],[rz]])*theta

    return r

'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass

'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass

if __name__ == "__main__":

    # Question 2.1

    # data = np.load('../data/some_corresp.npz')

    # pts1 = data['pts1']
    # pts2 = data['pts2']

    # im1 = plt.imread('../data/im1.png')
    # im2 = plt.imread('../data/im2.png')

    # M = np.max(im1.shape)

    # F = eightpoint(pts1, pts2,M) 
    # print(F)
    # np.savez('q2_1.npz', F=F, M=M)
    # helper.displayEpipolarF(im1, im2, F) 


    # Question 3.1

    # data = np.load('../data/some_corresp.npz')

    # pts1 = data['pts1']
    # pts2 = data['pts2']

    # im1 = plt.imread('../data/im1.png')
    # im2 = plt.imread('../data/im2.png')

    # M = np.max(im1.shape)

    # F = eightpoint(pts1, pts2,M) 

    # K = np.load('../data/intrinsics.npz')
    # K1 = K['K1']
    # K2 = K['K2']
    # E = essentialMatrix(F,K1,K2)
    # print(E)
    # np.savez('q3_1.npz', E=E)

    # Question 4.1

    # data = np.load('../data/some_corresp.npz')

    # pts1 = data['pts1']
    # pts2 = data['pts2']
    # #print(pts1)

    # im1 = plt.imread('../data/im1.png')
    # im2 = plt.imread('../data/im2.png')

    # im1 = plt.imread('../data/im1.png')
    # im2 = plt.imread('../data/im2.png')

    # M = np.max(im1.shape)

    # F = eightpoint(pts1, pts2,M) 

    # helper.epipolarMatchGUI(im1, im2, F)

    ### Check 4.1 file
    # data = np.load('q4_1.npz')

    # pts1 = data['pts1']
    # pts2 = data['pts2']
    # F = data['F']
    # print(pts1)
    # print(pts2)
    # print(F)


    # Question 5.1 Ransac

    # noisy = np.load('../data/some_corresp_noisy.npz')
    # pts1 = noisy['pts1']
    # pts2 = noisy['pts2']
    
    # im1 = plt.imread('../data/im1.png')
    # im2 = plt.imread('../data/im2.png')


    # K = np.load('../data/intrinsics.npz')
    # K1 = K['K1']
    # K2 = K['K2']

    # M = np.max(im1.shape)

    # # vary parameters below
    # tol = 0.8
    # nIters = 350

    # # ransac
    # F, inliers = ransacF(pts1, pts2, M, nIters, tol)
    

    # # no ransac
    # #F = eightpoint(pts1, pts2, M)  
    
    # helper.displayEpipolarF(im1, im2, F)

    # Q 5.2

    # rodriguez and inverse rodriguez, test matches matlab functions
    # r = [1,1,1]
    # R = rodrigues(r)
    # print(R)
    # r_vector = invRodrigues(R)
    # print(r_vector)

    # checking files are saved correctly

    # two_one = np.load("q2_1.npz")
    # F = two_one['F']
    # M = two_one['M']
    # print(F)
    # print(M)

    # three_one = np.load("q3_1.npz")
    # E= three_one['E']
    # print(E)

    # three_three = np.load("q3_3.npz")
    # M2= three_three['M2']
    # C2 = three_three['C2']
    # P = three_three['P']
    # print(M2)
    # print(C2)
    # print(P)

    # four_two = np.load("q4_2.npz")
    # F = four_two['F']
    # M1 = four_two['M1']
    # M2 = four_two['M2']
    # C1 = four_two['C1']
    # C2 = four_two['C2']
    # print(F)
    # print(M1)
    # print(M2)
    # print(C1)
    # print(C2)

    print('uncomment some above lines for testing')

