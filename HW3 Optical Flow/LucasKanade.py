import numpy as np
from scipy.interpolate import RectBivariateSpline
from numpy import matlib

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
	
    # Put your implementation here
    p = p0

    # x1,y1 are top left corner and x2,y2 are bottom right corner
    x1 = rect[0]
    x2 = rect[2]

    y1 = rect[1]
    y2 = rect[3]

    # dimensions of region, need as int, were float64
    width = int(x2-x1)
    height = int(y2-y1)

    # Splines for interpolation
    It_spline = RectBivariateSpline(np.linspace(0, It.shape[0], num= It.shape[0], endpoint = False),
                                    np.linspace(0, It.shape[1], num= It.shape[1], endpoint = False), It)
    
    It1_spline = RectBivariateSpline(np.linspace(0, It1.shape[0], num= It1.shape[0], endpoint =False),
                                     np.linspace(0, It1.shape[1], num= It1.shape[1], endpoint = False), It1)
    
    ### Iterate through while loop till iterations reached or found suitable p

    iteration = 1
    motion_convergence = 1
    # need meshgrid
    x, y = np.mgrid[x1:x2+1:width*1j, y1:y2+1:height*1j]
    

    #print('entered while loop')

    while num_iters>iteration and threshold<=motion_convergence:

        # Find gradients, flipped because rect cd system is different than image's
        pdx = It1_spline.ev( y+p[1], x+p[0], dy=1).flatten()
        pdy = It1_spline.ev( y+p[1], x+p[0], dx=1).flatten()

        # find pixels for template and pixels for current image region
        Template = It_spline.ev(y, x).flatten()
        Current = It1_spline.ev(y+p[1], x+p[0]).flatten()

        # Find A
        A = np.zeros((height*width, 2*height*width))
        # A[:, 0::2] = pdx.reshape(height*width, 1)
        # A[:, 1::2] = pdy.reshape(height*width, 1)
        # print('A before dot with idenitty')
        # print(A.shape)
        for i in range(height*width):
            A[i, 2*i] = pdx[i]
            A[i, 2*i+1] = pdy[i]
        A = np.dot(A, (matlib.repmat(np.eye(2), height*width, 1)))

        # Find b
        b = np.reshape(Template-Current, (height*width, 1))

        # find deltap
        dp = np.linalg.pinv(A).dot(b).T
        #print(p.shape, dp.shape)

        # Update variables
        motion_convergence = np.linalg.norm(dp)
        
        p = p + dp.flatten()
        
        iteration += 1

    return p



