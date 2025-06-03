# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

# Imports
import numpy as np
from matplotlib import pyplot as plt
from utils import integrateFrankot
import cv2
import skimage.color
from matplotlib import cm

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centerd on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """

    # get x and ys
    xm, ym = np.meshgrid(np.arange(res[0]), np.arange(res[1]))
    x = pxSize*(xm- res[0]/2) + center[0]
    y = pxSize*(ym- res[1]/2) + center[1]

    # get surdface values
    z = rad**2 - x**2 - y**2
    z0 = z< 0
    z[z0] = 0.0
    z = np.sqrt(z)

    # sphere locs
    sphere = np.stack((x, y, z), axis=2).reshape((res[0]*res[1], -1))
    sphere = (sphere.T /np.linalg.norm(sphere,axis=1).T).T
    
    # make image and set shadows
    image = np.dot(sphere, light).reshape((res[1], res[0]))
    image[z0] = 0.0

    return image


def loadData(path = "../data/"):

    

    I = None
    L = None
    s = None
    
    for n in range(1,8):
        # get imiage
        image = cv2.imread(path+"input_"+str(n)+".tif", -1) 
        # convert to xyz
        image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
        Irow = image[:,:,1]
        # set empty I and get shape
        if n ==1:
            h, w = Irow.shape
            s = (h,w)
            I = np.zeros((7, h*w))

        I[n-1,:] = np.reshape(Irow, (1,h*w))

    # get L
    L = np.load(path + "sources.npy")
    L = L.T


    return I, L, s




def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    
    B = np.linalg.inv(L@L.T) @L @I

    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''

    albedos = np.linalg.norm(B, axis=0)
    normals = B/albedos

    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `gray` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = np.reshape((albedos/np.max(albedos)), s)

    

    norm_normals = np.linalg.norm(normals, axis=0)
    normalized_normals = normals / norm_normals

    
    normalIm = normalized_normals.T.reshape((*s, 3))

    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """

    zx = np.reshape(normals[0,:]/ (-normals[2,:]),s)
    zy = np.reshape(normals[1,:]/ (-normals[2,:]),s)

    surface = integrateFrankot(zx, zy)

    return surface


def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """
    
    fig = plt.figure()
    x, y = np.meshgrid(np.arange(surface.shape[1]), np.arange(surface.shape[0]))

    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, surface, edgecolor='none', cmap=cm.coolwarm)
    plt.show()
    


if __name__ == '__main__':

    # Put your main code here
    
    ##### 

    # a thoery

    # b rendered spheres

    center = [0.0, 0.0, 0.0]
    radius = 0.75
    lights = [[1, 1, 1], [1, -1, 1], [-1, -1, 1]]/np.sqrt(3)
    pxSize = 7e-4 # in cm
    res = [3840, 2160]

    for i in range(len(lights)):
        # print three different light coords
        image = renderNDotLSphere(center, radius, lights[i], pxSize, res)
        scale_factor = .25  
        small_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
        cv2.imshow("image", small_image)
        cv2.waitKey(0)


    # c load function


    # d print svd

    I, L, s = loadData()

    U, S, VT = np.linalg.svd(I, full_matrices=False)
    print(S)


    # # e code and values
    B = estimatePseudonormalsCalibrated(I, L)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

    # f show albedo and normals
    plt.imshow(albedoIm, cmap='gray')
    plt.show()

    print('here')
    plt.imshow(normalIm, cmap='rainbow')
    plt.show()


    # # i
    surface = -1.0*estimateShape(normals, s)
    
    plotSurface(surface)
        
