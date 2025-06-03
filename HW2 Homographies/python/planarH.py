import numpy as np
import cv2
from opts import get_opts

def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
	# Lecture 8 slide 52

	
	num_of_coords = x1.shape[0]
	A = np.zeros((2*num_of_coords, 9))

	for i in range(num_of_coords):

		# individual x and y coords for x1 and x2
		x1_x = x1[i,0]
		x1_y = x1[i,1]
		x2_x = x2[i,0]
		x2_y = x2[i,1] 
		A[2*i] =   [-x1_x, -x1_y, -1, 0, 0, 0, x1_x*x2_x, x1_y*x2_x, x2_x]
		A[2*i+1] = [0, 0, 0, -x1_x, -x1_y, -1, x2_y*x1_x, x2_y*x1_y, x2_y]

    # SVD method

	U, S, Vt = np.linalg.svd(A)
	H2to1 = Vt[-1]/Vt[-1][-1]

	H2to1 = H2to1.reshape(3,3)

	return H2to1


def computeH_norm(x1, x2):
	#Q2.2.2
	#Compute the centroid of the points

	num_coords = x1.shape[0]

	centroid1 = np.mean(x1, axis=0)
	centroid2 = np.mean(x2, axis = 0)

    #Shift the origin of the points to the centroid
	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	max_distance1 = np.max(np.linalg.norm(x1 - centroid1, axis=1))
	max_distance2 = np.max(np.linalg.norm(x2 - centroid2, axis=1))

    
	scale1 = np.sqrt(2)/max_distance1
	scale2 = np.sqrt(2)/max_distance2

    #Similarity transform 1

	T1 = np.array([[scale1, 0, -scale1*centroid1[0]],
                  [0, scale1, -scale1*centroid1[1]],
                  [0, 0, 1]])
	
	x1_norm = np.dot(np.hstack((x1, np.ones((num_coords, 1)))), T1.T)[:, :2]

	#Similarity transform 2

	T2 = np.array([[scale2, 0, -scale2*centroid2[0]],
                  [0, scale2, -scale2*centroid2[1]],
                  [0, 0, 1]])

    
	x2_norm = np.dot(np.hstack((x2, np.ones((num_coords, 1)))), T2.T)[:, :2]

	#print(x1_norm)
	#print(x2_norm)

	#Compute homography
	H_norm = computeH(x1_norm,x2_norm)

	#Denormalization: switch tq and t2
	H2to1 = np.dot(np.dot(np.linalg.inv(T2), H_norm), T1)

	
	return H2to1




def computeH_ransac(locs1, locs2, opts):
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points
	max_iters = opts.max_iters  # the number of iterations to run RANSAC for
	inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

	# slide 44 lec 8 
	# loop through max times, get 4 random points from each, call homos, seeif they are within tol,
	# save new ones has less than most
	bestH2to1 = None
	inliers = [0]
	

	for n in range(max_iters):
		# pick points
		randlocs = np.random.choice(len(locs1), 4, replace=False)
		#print(randlocs)
		locs1_rand = locs1[randlocs]
		locs2_rand = locs2[randlocs]
		# print(locs1_rand)
		# print(locs2_rand)

        # apply transf
		H2to1 = computeH_norm(locs1_rand, locs2_rand)

		### loop through coords
		inlier_per_H = np.zeros((locs1.shape[0]))
		for i in range(locs1.shape[0]):
			
			# find x1s and x2s and predicitons, x y 1
			x1 = np.hstack((locs1[i], 1))
			x2 = np.hstack((locs2[i], 1))
			x2_guess = np.dot(H2to1, x1.T)

			x2_guess[0] = x2_guess[0]/x2_guess[2]
			x2_guess[1] = x2_guess[1]/x2_guess[2]

			### might need to change
			
			xerror = (locs2[i][0]-x2_guess[0])
			yerror = (locs2[i][1]-x2_guess[1])
			error = [xerror, yerror]
			diff = np.linalg.norm(error)

			# adjust inlier array
			if diff< inlier_tol:
				# add inlier to index locatin
				inlier_per_H[i] = 1

		
		# update per iteration in maxit
		if sum(inlier_per_H) > sum(inliers):
			inliers = inlier_per_H
			bestH2to1 = H2to1



	return bestH2to1, inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
	

	#Create mask of same size as template
	mask = np.ones_like(template) 

    #Warp mask by appropriate homography
	
	mask = cv2.warpPerspective(mask, H2to1, (img.shape[1], img.shape[0]))

    #Warp template by appropriate homography
	template = cv2.warpPerspective(template, H2to1, (img.shape[1], img.shape[0]))
    

    #Use mask to combine the warped template and the image
	composite_img = img * (1 - mask) + template * mask
	composite_img = composite_img.astype('uint8')
	#composite_img = cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB)
	
	
	return composite_img

