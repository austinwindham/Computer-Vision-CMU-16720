import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    #M_3x3 = np.vstack((M, [0, 0, 1]))

    # Initialize parameter vectors dp and p as arrays of zeros
    dp = np.zeros(6)
    p = M.flatten()

    # set conditions
    iteration = 1
    motion_convergence = 1

    # make mesh

    x1, y1, x2, y2 = 0, 0, It.shape[1], It.shape[0]
    x, y= np.meshgrid(np.arange(x1,x2), np.arange(y1,y2))

    # make splines

    It_spline = RectBivariateSpline(np.linspace(0, It.shape[0], num= It.shape[0], endpoint = False),
                                    np.linspace(0, It.shape[1], num= It.shape[1], endpoint = False), It)
    IT1_spline = RectBivariateSpline(np.linspace(0, It1.shape[0], num= It1.shape[0], endpoint =False),
                                     np.linspace(0, It1.shape[1], num= It1.shape[1], endpoint = False), It1)
    
    # get gradeitns
    pdx = It_spline.ev(y, x, dy=1).flatten()
    pdy = It_spline.ev(y, x, dx=1).flatten()

    # find A before while loop
    
    A = np.zeros((pdx.shape[0], 6))
    xflat = x.flatten()
    yflat = y.flatten()
    
    A[:, 0] = np.multiply(pdx, xflat)
    A[:, 1] = np.multiply(pdx, yflat)
    A[:, 2] = pdx
    A[:, 3] = np.multiply(pdy, xflat)
    A[:, 4] = np.multiply(pdy, yflat)
    A[:, 5] = pdy

    while num_iters > iteration and threshold <= motion_convergence:

        # apply tansform to get region
        xwarp = p[0]*x + p[1]*y + p[2]
        ywarp = p[3]*x + p[4]*y + p[5]

        common = (xwarp > 0) & (xwarp < x2) & (ywarp> 0) & (ywarp< y2)
        xwarp = xwarp[common]
        ywarp = ywarp[common]

        # get warped spline
        warp_spline = IT1_spline.ev(ywarp, xwarp)

        A_common = A[common.flatten()]

        # hessian 
        H = np.dot(A_common.T, A_common)
        b = warp_spline.flatten() - It[common].flatten()
        

        dp = np.linalg.inv(H)@A_common.T@ b

        # get dM
        M = np.vstack((np.reshape(p, (2, 3)), np.array([[0, 0, 1]])))
        dM = np.vstack((np.reshape(dp, (2, 3)), np.array([[0, 0, 1]])))
        dM[0, 0] += 1
        dM[1, 1] += 1
        
        # get M and update p
        M = M@np.linalg.inv(dM)

        p = M[:2, :].flatten()

        # update condiitons
        motion_convergence = np.linalg.norm(dp)
        iteration += 1

    M = M[:2, :]
    M = np.vstack((M, [0, 0, 1]))
    
    return M
