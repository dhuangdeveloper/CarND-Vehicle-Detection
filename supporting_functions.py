import numpy as np
import cv2
from skimage.feature import hog
from sklearn.utils import shuffle

def obtain_fold_group(group_id,K):
    # divide data into K folds and return the fold index each sample belongs to
    test_fold_index = np.zeros_like(group_id)
    fold_index_assignment = shuffle(np.arange(0, K), random_state=0)
    for c_id in np.unique(group_id):
        n_sample_in_group = np.sum(group_id == c_id)
        if c_id ==0:
            test_fold_index[group_id == c_id] = shuffle((np.floor_divide(np.arange(0, n_sample_in_group),
                                                                         np.ceil(n_sample_in_group/K))).astype(int),
                                                        random_state=0) 
        else:
            test_fold_index[group_id == c_id] = fold_index_assignment[(np.floor_divide(np.arange(0, n_sample_in_group), np.ceil(n_sample_in_group/K))).astype(int)]
    return test_fold_index 

def cv2_readRGB(image_name):
    """ read image """
    return cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)

def convert_RGB_to_color_space(image, color_space):
    """ convert RGB img to a new color space """
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(image) 
    return feature_image
               
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    """ extract hog features """
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  #block_norm = 'L2-Hys',
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       #block_norm = 'L2-Hys',
                       visualise=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
    """ extract binned colors """
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, nbins=32):   
    """ extract histogram features """    
    bins_range=(0, 256)
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def extract_features(images, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9, 
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    """ A wrapper to get features """    
    features = []
    # Iterate through the list of images
    for image in images:
        file_features = []
        # Read in each one by one
        # apply color conversion if other than 'RGB'
        feature_image = convert_RGB_to_color_space(image, color_space)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            file_features.append(
                np.ravel([get_hog_features(feature_image[:,:,w],
                                           orient, pix_per_cell, cell_per_block,
                                           vis=False, feature_vec=False)
                          for w in hog_channel]))
        features.append(np.concatenate(file_features))
    return features


# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


def search_cars(img, x_start_stop, y_start_stop, xy_window, xy_overlap,              
                clf, color_space, hog_channel, orient, pix_per_cell, cell_per_block, spatial_size,
                hist_bins, spatial_feat, 
                hist_feat, hog_feat): 
    """apply the vehicle/non-vehicle classifier to a set of windows defined by x_start_stop, y_start_stop, xy_window, xy_overlap
    
    This function precompute the Hog feature cross the entire applicable region
    """
    window_size = 64
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(np.round(xy_window[0]*(1 - xy_overlap[0])))
    ny_pix_per_step = np.int(np.round(xy_window[1]*(1 - xy_overlap[1])))
    # Compute the number of windows in x/y
    nx_buffer = np.int(np.round(xy_window[0]*(xy_overlap[0])))
    ny_buffer = np.int(np.round(xy_window[1]*(xy_overlap[1])))
    nx_steps = np.int(np.floor((xspan-nx_buffer)/nx_pix_per_step))
    ny_steps = np.int(np.floor((yspan-ny_buffer)/ny_pix_per_step) )


    # remove extra region that are not covered by window in xspan, yspan definition
    xspan = nx_pix_per_step*nx_steps+nx_buffer
    yspan = nx_pix_per_step*ny_steps+nx_buffer

    nblocks_per_window = (window_size // pix_per_cell) - cell_per_block + 1
    nx_cell_per_step = np.int(window_size//pix_per_cell*(1 - xy_overlap[0])) 
    ny_cell_per_step = np.int(window_size//pix_per_cell*(1 - xy_overlap[1]))

    # convert the window and step size setting to the resized image


    img_region_for_search = img[y_start_stop[0]:(y_start_stop[0]+yspan),x_start_stop[0]:(x_start_stop[0]+xspan)]
    xyspan_resized =(pix_per_cell* nx_cell_per_step * (nx_steps-1) + window_size,
                     pix_per_cell* ny_cell_per_step * (ny_steps-1) + window_size)
    x_scale = xspan / np.float32(xyspan_resized[0])
    y_scale = yspan / np.float32(xyspan_resized[1])
    img_region_resized = cv2.resize(img_region_for_search, xyspan_resized)


    # Initialize a list to append window positions to                            
    color_space_image = convert_RGB_to_color_space(img_region_resized, color_space)
    hog_featgures_whole_image = [get_hog_features(color_space_image[:,:,w],
                                                  orient, pix_per_cell, cell_per_block, feature_vec=False)
                                 for w in hog_channel]
        
    on_windows = []
    for xb in range(nx_steps):
        for yb in range(ny_steps):
            ypos = yb*nx_cell_per_step
            xpos = xb*ny_cell_per_step
            # Extract HOG for this patch

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = color_space_image[ytop:ytop+window_size, xleft:xleft+window_size]#, (64,64)

            # Get color features
            #spatial_features = bin_spatial(subimg, size=spatial_size)
            #hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            feature_list = []
            if spatial_feat == True:
                spatial_features = bin_spatial(subimg, size=spatial_size)
                feature_list.append(spatial_features)
            if hist_feat == True:
                # Apply color_hist()
                hist_features = color_hist(subimg, nbins=hist_bins)
                feature_list.append(hist_features)
            if hog_feat == True:
                hog_features = np.ravel([w[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() for w in hog_featgures_whole_image])

                feature_list.append(hog_features)
            test_features = np.concatenate(feature_list).reshape(1, -1)
            test_prediction = clf.predict(test_features)
            test_probability = clf.decision_function(test_features)
            if test_prediction == 1:
                x_window_start = x_start_stop[0]+np.int(xleft*x_scale)
                y_window_start = y_start_stop[0]+np.int(ytop*y_scale)
                on_windows.append(((x_window_start, y_window_start),
                                   (x_window_start+xy_window[0], y_window_start+xy_window[1]),
                                   test_probability))
    return on_windows                               


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        

        hog_features = [get_hog_features(feature_image[:,:,w],
                                                      orient, pix_per_cell, cell_per_block,
                                                      vis=False, feature_vec=True)
                                     for w in hog_channel]                
        img_features.append(np.ravel(hog_features))

    #9) Return concatenated array of features
    return np.concatenate(img_features)


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap < threshold] = 0
    # Return thresholded map
    return heatmap

def get_labeled_bboxes(labels):
    if labels[1]>=1:
        bbox = np.zeros((labels[1], 4))
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox[car_number-1,:] = np.array([np.min(nonzerox), np.min(nonzeroy), np.max(nonzerox), np.max(nonzeroy)])        
    else:
        bbox = None
    return bbox
    

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def get_intersection_area(box1, box2):
    tmp_img = np.zeros((np.maximum(box1[1][1], box2[1][1]), np.maximum(box1[1][0], box2[1][0])))
    tmp_img[box1[0][1]:box1[1][1], box1[0][0]:box1[1][0]] +=1
    tmp_img[box2[0][1]:box2[1][1], box2[0][0]:box2[1][0]] +=1
    return (tmp_img==2).sum()