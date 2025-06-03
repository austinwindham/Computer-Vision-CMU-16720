import numpy as np
import LucasKanadeAffine
import InverseCompositionAffine
import scipy
from scipy.interpolate import RectBivariateSpline


def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # put your implementation here
    # change mask to zeros, because motion points are ones
    mask = np.zeros(image1.shape, dtype=bool)

    # regular method
    #M = LucasKanadeAffine.LucasKanadeAffine(image1, image2, threshold, num_iters)
    # Inverse composition method
    M = InverseCompositionAffine.InverseCompositionAffine(image1, image2, threshold, num_iters)

    
    # splines and meshgrids like before
    It_spline1 = RectBivariateSpline(np.linspace(0, image1.shape[0], num= image1.shape[0], endpoint = False),
                                    np.linspace(0, image1.shape[1], num= image1.shape[1], endpoint = False), image1)
    It_spline2 = RectBivariateSpline(np.linspace(0, image2.shape[0], num= image2.shape[0], endpoint =False),
                                     np.linspace(0, image2.shape[1], num= image2.shape[1], endpoint = False), image2)
    
    x1, y1, x2, y2 = 0, 0, image1.shape[1], image1.shape[0]
    x, y= np.meshgrid(np.arange(x1,x2), np.arange(y1,y2))

    xwarp = M[0, 0]*x+ M[0, 1]*y + M[0, 2]
    ywarp = M[1, 0]*x + M[1, 1]*y+ M[1, 2]

    # find outside so you can set it to 0
    outside = (xwarp < 0) | (xwarp >= x2) | (ywarp < 0) | (ywarp >= y2)
    
    warpspline1 = It_spline1.ev(y,x)
    warpspline2 = It_spline2.ev(ywarp, xwarp)
    warpspline1[outside] = 0
    warpspline2[outside] = 0

    # make mask and dilate
    change = abs(warpspline2-warpspline1)
    locs = (change > tolerance) & (warpspline2 != 0)
    mask[locs] = 1
    #mask = scipy.ndimage.morphology.binary_erosion(mask)
    mask = scipy.ndimage.morphology.binary_dilation(mask, iterations=3)

    return mask
