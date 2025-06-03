import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
import multiprocessing
import os
#Import necessary functions
from loadVid import loadVid
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH




#Write script for Q3.1




### need to do harrypoterize for each frame
### loop through frames and call helper function that Harrypotterizes

def HarryPotter(cv_cover, book_frame,ar, opts):
    #print('hp')
    #print(frame)
    matches, locs1, locs2 = matchPics(cv_cover, book_frame, opts)
    x1 = locs1[matches[:,0], 0:2]
    x2 = locs2[matches[:,1],0:2]

    x1 = x1[:,[1,0]]
    x2 = x2[:,[1,0]]

    H2to1, inliers = computeH_ransac(x1,x2,opts)

    # need some type of resizing
    #hp_cover = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))
    # basic math to resize was 45330 207 433, got some black so cropped some y and had to change x too
    kfp_cropped = ar[50:310, 217:423, :]
    panda = cv2.resize(kfp_cropped, (cv_cover.shape[1], cv_cover.shape[0]))

    composite_img = compositeH(H2to1, panda, book_frame)



    return composite_img


###  Import videos, ar_source is kung fu panda
# ar_source = loadVid('../data/ar_source.mov')
# cv_cover = cv2.imread('../data/cv_cover.jpg')
# book = loadVid('../data/book.mov')
# opts = get_opts()

# print(ar_source.shape)
# print(cv_cover.shape)
# print(book.shape)

def process_frame(frame_data):
    try:
        print('yeet')
        cv_cover, book_frame, ar, opts, frame = frame_data
        print(frame)
        frame_image = HarryPotter(cv_cover, book_frame, ar, opts)
        return frame_image
    except Exception as e:
        # Log any exceptions to help with debugging
        print(f"Error in process_frame: {e}")
        return None  # Return a placeholder value to indicate an error

def main():
    # Load your data and define opts
    ar_source = loadVid('../data/ar_source.mov')
    cv_cover = cv2.imread('../data/cv_cover.jpg')
    book = loadVid('../data/book.mov')
    opts = get_opts()

    print(ar_source.shape)
    print(cv_cover.shape)
    print(book.shape)
    # Create a list of frame data, where each element is a tuple of (cv_cover, book_frame, ar, opts)
    frame_data_list = [(cv_cover, book[frame], ar_source[frame], opts, frame) for frame in range(ar_source.shape[0])]

    # Create a pool of worker processes
    num_processes = multiprocessing.cpu_count()  # Number of CPU cores
    p = multiprocessing.Pool(processes=num_processes)

    try:
        # Use p.map_async to process frame data asynchronously
        result = p.map_async(process_frame, frame_data_list)

        # Close the pool and wait for all processes to finish
        p.close()
        p.join()

        # Get the results from the async operation
        image_list = result.get()

        # Define output video details and save the video as you did before
        parent_directory = os.path.abspath(os.path.join(os.getcwd(), '..'))

        # Specify the output directory relative to the parent directory
        output_directory = os.path.join(parent_directory, 'result')

        # Define the output filename (you can change this as needed)
        output_filename = 'ar.avi'

        # Create the full path to the output file
        output_path = os.path.join(output_directory, output_filename)

        #output_file = 'ar.avi'
        codec = cv2.VideoWriter_fourcc(*'XVID')
        fps = 25
        frame_size = (image_list[0].shape[1], image_list[0].shape[0])
        out = cv2.VideoWriter(output_path, codec, fps, frame_size)

        for frame in image_list:
            out.write(frame)

        out.release()
    except Exception as e:
        # handle exceptions
        print(f"Error in main: {e}")

if __name__ == '__main__':
    main()