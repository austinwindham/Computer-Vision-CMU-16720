import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2
from scipy.ndimage import affine_transform

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    #M_3x3 = np.vstack((M, [0, 0, 1]))
    
    ### make it 3x3 at the end though based on piazza
    ### like before just affine transform now

    # switch M to p at the end
    dp = np.zeros(6)
    p = M.flatten()

    # set conditions
    iteration =1
    motion_convergence = 1

    # get shape
    x1, y1, x2, y2 = 0, 0, It.shape[1], It.shape[0]

    # It1 spline
    IT1_spline = RectBivariateSpline(np.linspace(0, It1.shape[0], num= It1.shape[0], endpoint =False),
                                     np.linspace(0, It1.shape[1], num= It1.shape[1], endpoint = False), It1)

    
    while num_iters>iteration and threshold<=motion_convergence:

        # mesh gird
        x, y= np.meshgrid(np.arange(x1,x2), np.arange(y1,y2))

        # apply warp to get new locs
        xwarp = p[0]*x + p[1]*y + p[2]
        ywarp = p[3]*x + p[4]*y + p[5]

        # get new area
        common = (xwarp > 0) & (xwarp < x2) & (ywarp> 0) & (ywarp< y2)
        xwarp = xwarp[common]
        ywarp = ywarp[common]
        x = x[common].flatten()
        y = y[common].flatten()

        # interpolate new it1 with warplocs
        warp_spline = IT1_spline.ev(ywarp, xwarp)

        # get gradients
        pdx = IT1_spline.ev(ywarp, xwarp, dy=1).flatten()
        pdy = IT1_spline.ev(ywarp, xwarp, dx=1).flatten()

        # Find A , no for loop with this technique
        A = np.zeros((pdx.shape[0], 6))
        
        A[:, 0] = np.multiply(pdx, x)
        A[:, 1] = np.multiply(pdx, y)
        A[:, 2] = pdx
        A[:, 3] = np.multiply(pdy, x)
        A[:, 4] = np.multiply(pdy, y)
        A[:, 5] = pdy

        # find hessian i think @ method is faster
        H = A.T@A

        # find b
        b = It[common].flatten() - warp_spline.flatten()
        #print(b.shape)

        # find dp
        dp = np.linalg.inv(H)@A.T@b
    
        # update p
        p += dp.flatten()

        # update coditions
        motion_convergence = np.linalg.norm(dp)
        iteration+=1

        
        #print('for the love of God')
    # shape p to M
    M = np.reshape(p, (2, 3))
    # make M 3x3
    M = np.vstack((M, [0, 0, 1]))

    return M
