# Import libraries
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from natsort import natsorted
from skimage import exposure
from collections import deque
import argparse
from moviepy.editor import VideoFileClip

# Load mtx, dist from pickle file saved
with open('calibration.p', mode='rb') as file:
    calibration = pickle.load(file)
    mtx = calibration["mtx"]
    dist = calibration["dist"]

# Functions
def read_img(img):
    return mpimg.imread(img)

def fix_distortion(img):
    return cv2.undistort(img, mtx, dist, None, mtx)

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def normalize(sobel_in):
    return np.uint8(255 * sobel_in / np.max(sobel_in))

def histo(img):
    return np.sum(img[img.shape[0]//2:,:], axis=0)

def eq_hist(img):
    return cv2.equalizeHist(img)

def sigmoid(img):
    return exposure.adjust_sigmoid(img)

def adjust_intensity(img):
    return exposure.rescale_intensity(img)

def adapthist(img):
    return exposure.equalize_adapthist(img)

def roi(img, vert = 'warped'):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #define vertices
    shape = img.shape
    if vert == 'warped':
        vertices = np.array([[(0,0),(shape[1],0),(shape[1],0),(6*shape[1]/7,shape[0]),
                          (shape[1]/7,shape[0]), (0,0)]],dtype=np.int32)
    elif vert == 'unwarped':
        # FROM https://github.com/wonjunee/Advanced-Lane-Finding/blob/master/Advanced-Lane-Finding-Submission.ipynb
        # Defining vertices for marked area
        imshape = img.shape
        left_bottom = (100, imshape[0])
        right_bottom = (imshape[1]-20, imshape[0])
        apex1 = (610, 410)
        apex2 = (680, 410)
        inner_left_bottom = (310, imshape[0])
        inner_right_bottom = (1150, imshape[0])
        inner_apex1 = (700,480)
        inner_apex2 = (650,480)
        vertices = np.array([[left_bottom, apex1, apex2, \
                          right_bottom, inner_right_bottom, \
                          inner_apex1, inner_apex2, inner_left_bottom]], dtype=np.int32)
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    
#     ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def sobel_helper(img, orient = 'x', k = 3):
    if orient == 'x':
        return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k)
    if orient =='y':
        return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=k)

def binary_output(output, thresh=(0, 255)):
    temp = np.zeros_like(output)
    temp[(output >= thresh[0]) & (output <= thresh[1])] = 1
    return temp


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):

    gray = grayscale(img)

    if orient =='x':
        sobel = sobel_helper(gray, 'x', sobel_kernel)
        abs_sobel = np.absolute(sobel)
    if orient =='y':
        sobel = sobel_helper(gray, 'y', sobel_kernel)
        abs_sobel = np.absolute(sobel)

    norm_sobel = normalize(abs_sobel)

    output = binary_output(norm_sobel, thresh)

    return output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

    gray = grayscale(img)

    sobelx = sobel_helper(gray, 'x', sobel_kernel)
    sobely = sobel_helper(gray, 'y', sobel_kernel)

    mag_sobel = np.sqrt((sobelx ** 2) + (sobely ** 2))

    norm_sobel = normalize(mag_sobel)

    output = binary_output(norm_sobel, mag_thresh)

    return output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):

    gray = grayscale(img)

    sobelx = sobel_helper(gray, 'x', sobel_kernel)
    sobely = sobel_helper(gray, 'y', sobel_kernel)

    grad_dir = np.arctan2(np.abs(sobely), np.abs(sobelx))

    output = binary_output(grad_dir, thresh)

    return output

def color_space(img, color = 'HLS', thresh=(0,255), selection = 0):
    
    if color == 'RGB':
        pass
    else:
        color_dict = {'HLS': 53, 'GRAY': 7, 'XYZ': 33, 'HSV': 41, 
                      'LAB': 45, 'LUV': 51, 'YUV': 83, 'YCrCb': 37}

        img = cv2.cvtColor(img, color_dict[color])
    
    channel_selection = selection
    channel = img[:,:,channel_selection]
    
    binary = binary_output(channel, thresh)
    
    return binary


def masking(img):
#     img = sigmoid(img)
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(20, 100))
#     grady = abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(0, 255))
#     mag_binary = mag_thresh(img, sobel_kernel=9, mag_thresh=(150, 200))
    dir_binary = dir_threshold(img, sobel_kernel=9, thresh=(0.2, 1.5))
    hls_s = color_space(img, color='HLS', thresh=(100,255), selection=2)
    hls_l = color_space(img, color='HLS', thresh=(120,255), selection=1)
    rgb_r = color_space(img, 'RGB', thresh=(150,255), selection = 0)
    rgb_g = color_space(img, 'RGB', thresh=(150,255), selection = 1)
    luv_l = color_space(img, 'LUV', thresh=(225,255), selection = 0)
    lab_b = color_space(img, 'LAB', thresh=(155,200), selection = 2)
#     yuv_y = color_space(img, 'YUV', thresh=(150,255), selection = 0)
#     hsv_v = color_space(img, 'HSV', thresh=(30,125), selection = 2)
#     hsv_s = color_space(img, 'HSV', thresh=(0,5), selection = 1)
#     hsv_h = color_space(img, 'HSV', thresh=(70,250), selection = 0)
#     ycr_y = color_space(img, 'YCrCb', thresh=(120,150), selection =1)
    combined_1 = np.zeros_like(dir_binary)
    combined_1[(((rgb_r==1) & (rgb_g==1)) & (hls_l==1)) & (((hls_s==1) | ((gradx==1) & (dir_binary==1))))]=1
    combined_2 = np.zeros_like(rgb_r)
#     combined_2[((rgb == 1) | (hls == 1)) | (gradx == 1) & ((dir_binary == 1))] = 1
    combined_2[(rgb_r == 1) | (hls_s == 1)] = 1
    combined_3 = np.zeros_like(dir_binary)
    combined_3[(luv_l == 1) | (lab_b == 1)] = 1

#     combined_2 = roi(combined_2, vertices)
    
    return combined_1

def warp(img):
    corners = np.float32([[190,720],[589,457],[698,457],[1145,720]])
    top_left=np.array([corners[0,0],0])
    top_right=np.array([corners[3,0],0])
    offset=[150,0]
    
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([corners[0],corners[1],corners[2],corners[3]])
#     src = np.float32([[570, 455], [790, 455], [1180, 720], [200, 720]])
    dst = np.float32([corners[0]+offset,top_left+offset,top_right-offset ,corners[3]-offset])    
#     dst = np.float32([[200, 0], [1180, 0], [1180, 720], [200, 720]])
#     if tobird:
    M = cv2.getPerspectiveTransform(src, dst)
#     else:
    Minv = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(img, M, img_size , flags=cv2.INTER_LINEAR)    
    return warped, M, src, dst, Minv

def pipeline(img):
    
#     img = read_img(img)
    
    img = fix_distortion(img)
    img = masking(img)
    img = roi(img, 'unwarped')
    # img = warp(img)[0]
    img, M, src, dst, Minv = warp(img)
#     img = roi(img)

    
    return img, M, src, dst, Minv


    
def draw_lane(left, right, color = 'green', linewidth = 3):
    
#     ploty = np.linspace(0, 719, num=720)
    
    plt.xlim(0, img.shape[1])
    plt.ylim(0, img.shape[0])
    
    plt.plot(left, ploty, color=color, linewidth=linewidth)
    plt.plot(right, ploty, color=color, linewidth=linewidth)
    plt.gca().invert_yaxis()
    
def fill_lane(image, left, right, l = True, r = True, color = 'green', linewidth = 3):
    
    # Create blank canvas
    zero_image = np.zeros_like(image).astype(np.uint8)
    zero_image = np.dstack((zero_image, zero_image, zero_image))
    
#     plt.plot(left, ploty, color=color, linewidth=linewidth)
#     plt.plot(right, ploty, color=color, linewidth=linewidth)
    
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
    
    pts_left = np.array([np.transpose(np.vstack([left, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    if l == True:
        cv2.polylines(zero_image, np.int_([pts_left]), isClosed = False, color=(0,0,255), thickness = 40)
    else:
        cv2.polylines(zero_image, np.int_([pts_left]), isClosed = False, color=(255,0,0), thickness = 40)
    if r == True:
        cv2.polylines(zero_image, np.int_([pts_right]), isClosed = False, color=(0,0,255), thickness = 40)
    else:
        cv2.polylines(zero_image, np.int_([pts_right]), isClosed = False, color=(255,0,0), thickness = 40)
    
    cv2.fillPoly(zero_image, np.int_([pts]), (0, 255, 0))
    
    return zero_image
    
def unwarp(image, Minv):
    
    image = cv2.warpPerspective(image, Minv, (image.shape[1], image.shape[0]))
    
    return image

def combine_result(original, image):
    
    result = cv2.addWeighted(original, 1, image, 0.3, 0)
    
    return result
    
def add_results(image):
    

    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Radius of Curvature:"
    cv2.putText(image, text, (50,50), font, 1, (255,255,255), 2)
    text = "How far from center"
    cv2.putText(image, text, (50,100), font, 1, (255,255,255), 2)
    
    return image

def fit_polynomial(poly, y, marg = 0):
    
    poly = np.asarray(poly)
    y = np.asarray(y)
    marg = np.asarray(marg)
    return ((poly[0] * y ** 2) + (poly[1] * y) + poly[2] + marg)

class Line():
    
    def __init__(self, side):
        
        # was the line detected in the last iteration?
        self.detected = False  
        
        # was their enough points found through line search
        self.trigger = False
        
        # Set which side the Line object is
        self.side = side
        
        # Count how many values in history list
        self.history = 0
        
        # fit from previous iteration
        self.fit = None
        
        # fitted x values from previous iteration
        self.fitx = None
        
        # recent list of fit values
        self.fit_list = deque(maxlen = 10)
        
        # recent list of x values
        self.fitx_list = deque(maxlen = 10)
        
        # average of last 10 fits
        self.avg_fit = None#np.mean(self.fit_list, axis=0)
        
        # average of last 10 x values
        self.avg_fitx = None#np.mean(self.fitx_list, axis=0)
        
        # intercept top
        self.int_top = None
        
        # intercept bot
        self.int_bot = None
        
        # list of top intercepts
        self.int_top_list = deque(maxlen = 10)
        
        # list of bot intercepts
        self.int_bot_list = deque(maxlen = 10)
        
        # average of top intercepts
        self.avg_top = None#np.mean(self.int_top_list, axis=0)
        
        # average of bot intercepts
        self.avg_bot = None#np.mean(self.int_bot_list, axis=0)
    
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        
        # Keep number of total frames
        self.total_count = 0
        
        # Keep number of good frames
        self.good_count = 0
        
        # Keep number of bad frames
        self.bad_count = 0
        
    def blind_find_lane(self, image):
        
        side = self.side
        
        out_img = np.dstack((image, image, image))*255

        # Create histogram of processed image
        histogram = np.sum(image[image.shape[0]//2:,:], axis=0)

        # Find midpoints, and peak points of left and right half of the histogram
        midpoint = np.int(histogram.shape[0]/2)
        
        if self.side == 'left':
            x_base = np.argmax(histogram[:midpoint])
        elif self.side == 'right':
            x_base = np.argmax(histogram[:midpoint]) + midpoint
        else:
            raise ValueError

        # Choose the number of sliding windows
        nwindows = 9

        # Set height of windows
        window_height = np.int(image.shape[0]/nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        x_current = x_base

        # Set the width of the windows +/- margin
        margin = 100

        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Create empty lists to receive lane pixel indices
        lane_inds = []

        for window in range(nwindows):

            # Identify window boundaries in x and y (and right and left)
            win_y_low = image.shape[0] - (window+1)*window_height
            win_y_high = image.shape[0] - window*window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0,255,0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

            # Append these indices to the lists
            lane_inds.append(good_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))

        # Concatenate the arrays of indices
        lane_inds = np.concatenate(lane_inds)
#         print (self.side, len(lane_inds))

        # Extract line pixel positions
        x = nonzerox[lane_inds]
#         print (self.side, len(x))

        y = nonzeroy[lane_inds]
    
        ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
        
    
        if len(x) < 100:
            
            self.fit = None
            self.fitx = None
            self.int_bot = None
            self.int_top = None
            self.trigger = False
        
        else:

            # Fit a second order polynomial
            self.fit = np.polyfit(y, x, 2)
            self.fitx = self.fit_polynomial(self.fit, ploty)
            self.int_bot = self.fit_polynomial(self.fit, 720)
            self.int_top = self.fit_polynomial(self.fit, 0)
            self.trigger = True

        return out_img

    def exp_find_lane(self, image, fit):
        
#         side = self.side
        
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        
        lane_inds = ((nonzerox > self.fit_polynomial(fit, nonzeroy, marg = -margin)) & (nonzerox < self.fit_polynomial(fit, nonzeroy, marg = margin)))

        # Again, extract left and right line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        # Generate x and y values for plotting
        ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
        
        if len(x) < 100:
            
            self.fit = None
            self.fitx = None
            self.int_bot = None
            self.int_top = None
            self.trigger = False
        
        else:

            # Fit a second order polynomial
            self.fit = np.polyfit(y, x, 2)
            self.fitx = self.fit_polynomial(self.fit, ploty)
            self.int_bot = self.fit_polynomial(self.fit, 720)
            self.int_top = self.fit_polynomial(self.fit, 0)
            self.trigger = True

        out_img = fill_lane(image, self.fitx-margin, self.fitx+margin)
        
        return out_img
    
    def fit_polynomial(self, poly, y, marg = 0):
        poly = np.asarray(poly)
        y = np.asarray(y)
        marg = np.asarray(marg)
        return ((poly[0] * y ** 2) + (poly[1] * y) + poly[2] + marg)
    
    def calc_curvature(self, y, x):
    
        # Define y-value for location of curvature,  I chose middle y-value that corresponds
        # middle of the image
        y_eval = np.max(y) / 2

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit converted x,y values to polynomial
        fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)

        # Calculate the curvature
        curverad = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])

    #     print (curverad)

        return curverad

    def find_center(self, left, right):

#         left = left[0]*720**2 + left[1]*720 + left[2]
#         right = right[0]*720**2 + right[1]*720 + right[2]

    #     center = (1.5 * left - right) / 2
        center = 640 - ((right + left) / 2)

        xm_per_pix = 3.7/700 

        ret = center * xm_per_pix

        return ret

    
def sanity_check(left_fitx, right_fitx):
    
    left_lane_status = False
    right_lane_status = False
    
    if (((max(abs(right_fitx))-max(abs(left_fitx))) > 600) & ((max(abs(right_fitx)) - max(abs(left_fitx))) < 800)):
        if (((min(abs(right_fitx))-min(abs(left_fitx))) > 600) & ((min(abs(right_fitx))-min(abs(left_fitx))) < 800)):
#             print('lane wide enough')
#         left_lane_status = True
#         right_lane_status = True
    
            if (max(abs(left_fitx)) - min(abs(left_fitx))) < 300:
#                 print('left lane seems ok')
                left_lane_status = True
            if (max(abs(right_fitx)) - min(abs(right_fitx))) < 300:
#                 print('right lane seems ok')
                right_lane_status = True
            
    return left_lane_status, right_lane_status

global left, right
left = Line('left')
right = Line('right')            

def final_process(img):

    # if 'left' not in locals():
    #     global left
    #     left=Line('left')
    # if 'right' not in locals():
    #     global right
    #     right=Line('right')


    original = img
    original = fix_distortion(original)

    ploty = np.linspace(0, original.shape[0]-1, original.shape[0] )

    processed, M, src, dst, Minv = pipeline(img)

    if left.detected == False:
        left_img = left.blind_find_lane(processed)
    else:
        left_img = left.exp_find_lane(processed, left.fit)
        
    if right.detected == False:
        right_img = right.blind_find_lane(processed)
    else:
        right_img = right.exp_find_lane(processed, right.fit)


    def check_line(line):
        if line.history < 10:
            return True
        else:
            if (abs(line.int_bot - line.avg_bot) <= 100) & (abs(line.int_top - line.avg_top) <= 100):
                return True
    #             elif line
            else:
                return False



    for line in [left, right]:
        
        
        if line.trigger == True:
        
            if line.history < 10:

                line.fit_list.append(line.fit)
                line.fitx_list.append(line.fitx)
                line.int_top_list.append(line.int_top)
                line.int_bot_list.append(line.int_bot)
                line.avg_fit = np.mean(line.fit_list, axis=0)
                line.avg_fitx = np.mean(line.fitx_list, axis=0)
                line.avg_top = np.mean(line.int_top_list, axis=0)
                line.avg_bot = np.mean(line.int_bot_list, axis=0)
                line.history = len(line.fit_list)
                line.good_count += 1
                line.detected = True
                
            else:
                
                if check_line(line) == True:
                    
                    line.fit_list.append(line.fit)
                    line.fitx_list.append(line.fitx)
                    line.int_top_list.append(line.int_top)
                    line.int_bot_list.append(line.int_bot)
                    line.avg_fit = np.mean(line.fit_list, axis=0)
                    line.avg_fitx = np.mean(line.fitx_list, axis=0)
                    line.avg_top = np.mean(line.int_top_list, axis=0)
                    line.avg_bot = np.mean(line.int_bot_list, axis=0)
                    line.history = len(line.fit_list)
                    line.good_count += 1
                    line.detected = True
                    
                else:
                    
                    line.bad_count += 1
                    line.detected = False
                
        else:
            
            line.bad_count += 1
            line.detected = False
        

    left.total_count += 1
    right.total_count += 1

    if left.detected == True:
        display_left = left.fitx
        center_left = left.int_bot
    else:
        display_left = left.avg_fitx
        center_left = left.avg_bot

    if right.detected == True:
        display_right = right.fitx
        center_right = right.int_bot
    else:
        display_right = right.avg_fitx
        center_right = right.avg_bot

    filled_image = fill_lane(processed, display_left, display_right, left.detected, right.detected)    

    unwarped = unwarp(filled_image, Minv)

    result = combine_result(original, unwarped)

    curvature_left = left.calc_curvature(ploty, display_left)
    curvature_right = left.calc_curvature(ploty, display_right)

    center = left.find_center(center_left, center_right)

    font = cv2.FONT_HERSHEY_SIMPLEX

    text = "Radius of Left Curvature: {:.2f} m".format(curvature_left)
    cv2.putText(result, text, (50,50), font, 1, (255,255,255), 2)
    text = "Radius of Right Curvature: {:.2f} m".format(curvature_right)
    cv2.putText(result, text, (50,100), font, 1, (255,255,255), 2)
    if center < 0:
        text = "{:.2f} m left of center".format(abs(center))
    else:
        text = "{:.2f} m right of center".format(abs(center))
    cv2.putText(result, text, (50,150), font, 1, (255,255,255), 2)
    if left.detected == False:
        text = "Left lane: Lost"
        cv2.putText(result, text, (50,200), font, 1, (255,0,0), 2)
    else:
        text = "Left lane: Found"
        cv2.putText(result, text, (50,200), font, 1, (0,255,0), 2)

    if right.detected == False:
        text = "Right lane: Lost"
        cv2.putText(result, text, (50,250), font, 1, (255,0,0), 2)
    else:
        text = "Right lane: Found"
        cv2.putText(result, text, (50,250), font, 1, (0,255,0), 2)
    text = "Frame: {}".format(left.total_count)
    cv2.putText(result, text, (1000,50), font, 1, (255,255,255), 2)


    return result

# def out_process(img, line_1='left', line_2='right'):
#     left = line_1
#     right= line_2

#     img = final_process(img)

#     return img



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='lane finder')
	parser.add_argument('file', type=str, help='Path to image file')
	args = parser.parse_args()

	# left = Line('left')
	# right = Line('right')

	if ('.jpg' or '.jpeg' or '.png') in args.file:
		file = read_img(args.file)
		# _, M, src, dst, Minv = warp(file)

		print(args.file)
		plt.imshow(final_process(file))
		plt.show()

	elif '.mp4' in args.file:
		output = 'find_lane_result.mp4'
		clip1 = VideoFileClip(args.file)
		output_clip = clip1.fl_image(final_process)
		output_clip.write_videofile(output, audio=False)