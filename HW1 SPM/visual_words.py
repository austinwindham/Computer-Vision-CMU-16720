import os, multiprocessing
from multiprocessing import Pool
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
import sklearn.cluster
import opts
from opts import get_opts




def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    filter_scales = opts.filter_scales
    # ----- TODO -----

    #Turn grayscale image into image with three channels
    if len(img.shape) ==2:
        img = np.stack((img,img,img),axis = -1)
    
    if len(img.shape) == 3 and img.shape[2] ==1:
        img = np.repeat(img, 3, axis = -1)

    # Convert range to [0,1] floating point if necessary
    # normalize method or /255 method, went with the latter
    if img.max()>1:
        img = img/255


    # convert to Lab color space
    img = skimage.color.rgb2lab(img)

    #create filter_responses, loop thourgh sigmas within rgb channels
    filter_responses = None
    Logstack = None
    
    for f in filter_scales:
        Gaus = scipy.ndimage.gaussian_filter(img,sigma = (f,f,0), order = 0)
        Gaus1 = scipy.ndimage.gaussian_filter(img,sigma = (f,f,0), order = 0)
        Logstack = None
        for i in range(3):
            Log = scipy.ndimage.gaussian_laplace(img[:,:,i],sigma = f)
            if Logstack is None:
                Logstack = Log
            else:
                Logstack = np.dstack((Logstack, Log))
        ### Question about derivative
        Dogx = scipy.ndimage.gaussian_filter(img,sigma = (f,f,0), order = (0,1,0))
        Dogy = scipy.ndimage.gaussian_filter(img,sigma = (f,f,0), order = (1,0,0))
        if filter_responses is None:
            filter_responses = np.dstack((Gaus,Logstack,Dogx,Dogy))
        else:
            filter_responses = np.dstack((filter_responses, Gaus,Logstack,Dogx,Dogy))
        
    

                

    return filter_responses

def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''
    ### Changed that args to opts, img, is that what I should have done
    # ----- TODO -----
    ### Question about img
    img_file = args
    opts = get_opts()
    
    ### Load Image based off main.py
    img_path = join(opts.data_dir, img_file)
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    filter_responses = extract_filter_responses(opts, img)

    # Get alpha pixels
    [height,width] = [int(img.shape[0]), int(img.shape[1])]
    # Old Mehtod RandomArray = np.random.choice(height*width, opts.alpha)
    RandomRow = np.random.choice(height, opts.alpha)
    RandomColumn = np.random.choice(width, opts.alpha)
    # Part of old mehtodRandomRow = (RandomArray/width).astype(int)
    # sameRandomColumn = (RandomArray%width).astype(int)
    
    
    alpha_pixel_responses = filter_responses[RandomRow,RandomColumn,:]
    #print(alpha_pixel_responses.shape)

    # Save this into a new temporary folder
    alpha_img = img_file.replace('/','_').split('.')[0]
    np.save(os.path.join('temporary/',alpha_img+'.npy'),alpha_pixel_responses)
    
    return


def compute_dictionary(opts, n_worker=16):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    # ----- TODO -----
    # create folder
    if os.path.exists('temporary'):
        print('temporary file already made')
    else:
        os.mkdir('temporary')

    #Do multiprocessing to run compute one images simultaneously
    # extract_filter_responses needs opts and images
    
    p = multiprocessing.Pool(processes=n_worker)
    
    p.map_async(compute_dictionary_one_image, train_files)
    p.close()
    p.join()

    first_seq = True

    for f in train_files:
        #compute_dictionary_one_image(f)
        fn = f.replace('/','_').split('.')[0]
        #print(fn)
        file = np.load('temporary/'+fn+'.npy')
        if first_seq:
            alpha_filter_responses = file
            first_seq= False
        else:
            alpha_filter_responses = np.vstack((alpha_filter_responses, file))

    #print(alpha_filter_responses)
    
    kmeans = sklearn.cluster.KMeans(n_clusters=opts.K).fit(alpha_filter_responses)
    dictionary = kmeans.cluster_centers_
    print(dictionary.shape)

    ## example code snippet to save the dictionary
    np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO -----
    ### dictionary is wordsx number of filters
    ### filtered response is hxwxfilters
    ### scipy command: arrays have to have the same number of rows, so reshape image to number of filters
    ### turn filtered resposne into one long array
    filter_scales = opts.filter_scales
    response = extract_filter_responses(opts,img)
    #filternums = dictionary.shape[1]
    filternums = 3*len(filter_scales)*4
    reshaped_response = response.reshape(response.shape[0]*response.shape[1], filternums)
    # print(reshaped_response.shape)
    # print(dictionary.shape)
    euclidean_distance = scipy.spatial.distance.cdist(reshaped_response,dictionary,'euclidean')
    min_dis = np.argmin(euclidean_distance, axis=-1)
    wordmap = min_dis.reshape(response.shape[0],response.shape[1])

    return wordmap



