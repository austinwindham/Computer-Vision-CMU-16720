# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface 
from utils import enforceIntegrability
from matplotlib import pyplot as plt


def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """

    U, S, VT = np.linalg.svd(I, full_matrices=False)

    S[3:] = 0.0
    B = VT[0:3, :]
    L = U[0:3, :]

    return B, L


if __name__ == "__main__":

    # Put your main code here
    
    # a thoery

    # b show albedo and normals

    I, L0, s = loadData()
    B, L = estimatePseudonormalsUncalibrated(I)
    
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

    plt.imshow(albedoIm, cmap='gray')
    plt.show()

    plt.imshow(normalIm, cmap='rainbow')
    plt.show()


    # # c L values
    print('original')
    print('')
    print(L0)
    print('')
    print('factorized')
    print(L)
    np.set_printoptions(precision=4, suppress=True)
    print('')
    # Print fromatted aray L
    print(L)


    # d plot bad surface

    I, L0, s = loadData()
    B, L = estimatePseudonormalsUncalibrated(I)
    albedos, normals = estimateAlbedosNormals(B)
    surface = -1.0*estimateShape(normals, s)    
    plotSurface(surface)

    # e add integrability
    I, L0, s = loadData()
    B, L = estimatePseudonormalsUncalibrated(I)

    Nt = enforceIntegrability(B,s)
    albedos, normals = estimateAlbedosNormals(Nt)
    surface = -1.0*estimateShape(Nt, s)    
    
    plotSurface(surface)



    # f add bas relief
    mu = 0.5
    v = 1.5
    lamb = 0.1 
    G = np.asarray([[1, 0, 0], [0, 1, 0], [mu, v, lamb]])
    B = np.linalg.inv(G.T).dot(B)

    albedos, normals = estimateAlbedosNormals(B)
    normals = enforceIntegrability(normals, s)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

    surface= -1.0*estimateShape(normals, s)

    
    
    
    plotSurface(surface)
