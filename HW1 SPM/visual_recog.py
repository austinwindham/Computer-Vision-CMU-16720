import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    
    # ----- TODO -----
    # make histogram
    # histogram numpy needs flattend array
    # has to be L1 normalized, do so by dividing by the sum

    # Reminder note: bins range is K+1, to include K
    bins_input = np.arange(0,K+1)
    
    h, b = np.histogram(wordmap.flatten(), bins= bins_input )
    hist = h/np.sum(h)

    return hist

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    # ----- TODO -----
    # Default L was 1
    # L + 1 layer bc layer 0 exists
    # layer 0 and 1 same weight l gets half weight of l+1
    # popular to set layer 0 and 1 to 2^-L and the rest to 2^l-L-1
    # Tactic: loop through layers, create weights and division intervals
    # loop through division intervals double loop: one for x and one for y, and call get feature from wordmap
    # hstack them and them normalize outside all the loops

    hist_all = np.array([])

    for i in range(L+1):
        # make weights
        if L <2:
            weight = pow(2,-1*L)
        else:
            weight = pow(2,i-L-1)
        
        # calc nummber of divisions for this interval
        div_num = pow(2,L)

        # numpy arrays in x and y directin for width and height, if wrodmap heigh is 10 and we need 4 divisions,
        # 0 2.5 5 7.5 8, but can't do half division, so range from 0 to 11 with 2.5 intervals, but then round
        width_divisions = np.round(np.arange(0, wordmap.shape[1]+1,wordmap.shape[1]/div_num)).astype(int)
        height_divisions = np.round(np.arange(0, wordmap.shape[0]+1,wordmap.shape[0]/div_num)).astype(int)

        # loop thourgh x and y and call getfeature from wordmap
        for x in range(div_num):
            current_x = width_divisions[x]
            next_x = width_divisions[x+1]

            for y in range(div_num):
                current_y = height_divisions[y]
                next_y = height_divisions[y+1]

                # section wordmap and run smaller function and multiply by weight
                sub_section = wordmap[current_y:next_y , current_x:next_x]
                hist_section = get_feature_from_wordmap(opts, sub_section)
                hist_section = hist_section*weight

                hist_all = np.hstack((hist_all, hist_section))

    hist_all = hist_all/np.sum(hist_all)

    return hist_all
    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K*(4^L-1)/3)
    '''

    # ----- TODO -----
    # loads image
    # extracts wordmap
    # computes SPM
    # returns computed feature
    # do same way as in main.py
    img_path = join(opts.data_dir, img_path)
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255

    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    feature = get_feature_from_wordmap_SPM(opts, wordmap)

    return feature

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))
    print(dictionary.shape)

    # ----- TODO -----
    # got to save using the following example
    # need to run get geatures for all images
    # only thing missing from the example save code
    # can't get multiprocessing. try later if I have time
    features = np.array([])
    for n in train_files:
        one_feature = get_image_feature(opts,n, dictionary)
        if features.size ==0:
            features = one_feature
        else:
            features = np.vstack((features, one_feature))
    

    ## example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
         features=features,
         labels=train_labels,
         dictionary=dictionary,
         SPM_layer_num=SPM_layer_num,
     )

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * hist_dist: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    # sum of minimum value of each corresponding bins

    min_value = np.minimum(word_hist, histograms)
    hist_dist = np.sum(min_value, axis = 1)

    return hist_dist
  
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    # get all stuff out of trained system dictionaryesque file
    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']
    features = trained_system['features']
    labels = trained_system['labels']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)


    # ----- TODO -----
    # npz is essentially a dictionary
    # return confusion matrix and accuracy
    # make 8x8 matrix where C(i,j) corresponds to instances of i predictd as j
    # make blank 8x8 matrix first for conf
    # loop through files and compute distance for every image and return label of closest training image
    # this should be max simularity then do labels[most similar] and hope it matches to the ttest label, at least 50 %
    # acc = trace/sum of c
    conf = np.zeros((8,8), dtype=int)

    


    for f in test_files:
        img_path = join(opts.data_dir, f)
        img = Image.open(img_path)
        img = np.array(img).astype(np.float32)/255

        # make wordmap
        wordmap = visual_words.get_visual_words(opts, img, dictionary)
        current_features = get_feature_from_wordmap_SPM(opts, wordmap)
        max_simularity = np.argmax(distance_to_set(current_features, features))

        predicted = labels[max_simularity]
        actual = test_labels[test_files.index(f)]
        conf[actual, predicted] +=1


    accuracy = np.trace(conf)/np.sum(conf)
    return conf, accuracy

