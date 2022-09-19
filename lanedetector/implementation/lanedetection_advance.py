## ADVANCE LANE DETECTION SCRIPT ## 
# @AUTHOR FAZLI FARUK OKUMUS
# @AUTHOR MEVLUDE TIGRE  
## ADVANCE LANE DETECTION SCRIPT ## 


## IMPORT PART ##
from turtle import right
from typing import final
import matplotlib.pyplot as plt
import cv2
import numpy as np

## THRESHOLDING ##
def threshold(channel, thresh=(128,255), thresh_type=cv2.THRESH_BINARY,plot=False):
    _,threshold_image=cv2.threshold(channel, thresh[0], thresh[1], thresh_type)
    if plot == True:
        cv2.imshow("Frame After Thresholding", threshold_image)
    return threshold_image

## NOISE REDUC WITH GAUSS KERNEL ##
def processGauss(frame):
    noisereduc_image = cv2.GaussianBlur(frame, (7, 7), 1.41)
    return noisereduc_image

    
## CONVERT HLS ## 
def processHLS(frame):
    hls_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    return hls_image


## CONVERT GRAY ## 
def processGray(noisereduc_image):
    gray_image = cv2.cvtColor(noisereduc_image, cv2.COLOR_BGR2GRAY)
    return gray_image

## CANNY EDGE DETECTION ##
def processEdge(gray_image,plot=True):
    edge_image = cv2.Canny(gray_image, 150, 200)
    if plot==True:
        cv2.imshow("After edge detection",edge_image)
    return edge_image

## CANNY EDGE DETECTION ##
def binaryProcess(thresholdS_image, thresholdR_image,edge_image,plot=False):
    rs_binary = cv2.bitwise_and(thresholdS_image, thresholdR_image)
    preprocess_image=cv2.bitwise_or(rs_binary, edge_image.astype(np.uint8))    
    if plot==True:
        cv2.imshow("Frame after preprocessing",preprocess_image)
    return preprocess_image    

## DEFINE ROI AND CROP THE IMAGE CAlCULATE TRANFORMATION MATRICES ## 
def calcTM(preprocess_image,offset=120,point_offset=120,y_offset=100,toright=20,height_crop=50,plot=False):

    height = preprocess_image.shape[0]
    width = preprocess_image.shape[1]
    yf=(height/2)+y_offset
    xcenter=width/2

    vertices = (offset+toright,height-height_crop), (xcenter-point_offset, yf), (xcenter+point_offset,yf), (width-offset+toright,height-height_crop) # vertices first width after height
    vertices = np.array([vertices],np.int32)
    roi = np.float32([(offset+toright,height-height_crop), (xcenter-point_offset, yf), (xcenter+point_offset,yf), (width-offset+toright,height-height_crop)])
    desired= np.float32([(offset,width),(offset,0),(height-offset, 0),(height-offset,width)])
    M = cv2.getPerspectiveTransform(roi, desired)
    Minv = cv2.getPerspectiveTransform(desired,roi)
    mask = np.zeros_like(preprocess_image)
    mask=cv2.fillPoly(mask,vertices,255)
    roi_image = cv2.bitwise_and(preprocess_image,mask)
    if plot == True:
        cv2.imshow("Frame After preprocessing and ROI", roi_image)
    return roi_image,roi,desired,M,Minv

## DO PERSPECTIVE TRANSFORMATION ##    
def perspectiveTransform(preprocess_image,M,plot=False):
    frame_size = (preprocess_image.shape[0], preprocess_image.shape[1])
    warp_image = cv2.warpPerspective(preprocess_image, M, frame_size, flags=cv2.INTER_LINEAR)
    if plot==True:
        cv2.imshow("Frame after Perspective Transform",warp_image)
    return warp_image

## DETECT LANE WTIH HISTOGRAM ## 
def lane_histogram(img, height_start=800, height_end=1250,plot=False):
    histogram = np.sum(img[int(height_start):int(height_end),:], axis=0)
    midpoint = int(histogram.shape[0]/2)
    peak_left = np.argmax(histogram[:midpoint])
    peak_right = np.argmax(histogram[midpoint:]) + midpoint
    if plot==True:
      plt.plot(histogram)
      plt.title("Histogram Peaks")
      plt.pause(0.0000001)
      plt.clf()
      #plt.show()

    return histogram,peak_left,peak_right

## DETECT WHITE PIXELS BY USING SLIDING WINDOW TECHNIQUE ##
def lane_polys(warped_frame,margin=160,no_of_windows=10,minpix=80):

    frame_sliding_window = warped_frame.copy()
    # Set the height of the sliding windows
    window_height = int(warped_frame.shape[0]/no_of_windows)       
 
    # Find the x and y coordinates of all the nonzero 
    # (i.e. white) pixels in the frame. 
    nonzero = warped_frame.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1]) 
         
    # Store the pixel indices for the left and right lane lines
    left_lane_inds = []
    right_lane_inds = []
         
    # Current positions for pixel indices for each window,
    # which we will continue to update
    _, leftx_base, rightx_base = lane_histogram(warped_frame)
    leftx_current = leftx_base
    rightx_current = rightx_base
 
    for window in range(no_of_windows):
       
      # Identify window boundaries in x and y (and right and left)
      win_y_low = warped_frame.shape[0] - (window + 1) * window_height
      win_y_high = warped_frame.shape[0] - window * window_height
      win_xleft_low = leftx_current - margin
      win_xleft_high = leftx_current + margin
      win_xright_low = rightx_current - margin
      win_xright_high = rightx_current + margin
      cv2.rectangle(frame_sliding_window,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (255,255,255), 2)
      cv2.rectangle(frame_sliding_window,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (255,255,255), 2)
 
      # Identify the nonzero pixels in x and y within the window
      good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (
                           nonzerox < win_xleft_high)).nonzero()[0]
      good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (
                            nonzerox < win_xright_high)).nonzero()[0]
                                                         
      # Append these indices to the lists
      left_lane_inds.append(good_left_inds)
      right_lane_inds.append(good_right_inds)
         
      # If you found > minpix pixels, recenter next window on mean position
      if len(good_left_inds) > minpix:
        leftx_current = int(np.mean(nonzerox[good_left_inds]))
      if len(good_right_inds) > minpix:        
        rightx_current = int(np.mean(nonzerox[good_right_inds]))
                     
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
 
    # Extract the pixel coordinates for the left and right lane lines
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds] 
    righty = nonzeroy[right_lane_inds]
 
    # Fit a second order polynomial curve to the pixel coordinates for
    # the left and right lane lines
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)    
             
    return left_fit, right_fit,frame_sliding_window

def lane_line(warped_frame,left_fit, right_fit,margin=160,no_of_windows=10,minpix=80):

 
    # Find the x and y coordinates of all the nonzero 
    # (i.e. white) pixels in the frame.         
    nonzero = warped_frame.nonzero()  
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
         
    # Store left and right lane pixel indices
    left_lane_inds = ((nonzerox > (left_fit[0]*(
      nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (
      nonzerox < (left_fit[0]*(
      nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(
      nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (
      nonzerox < (right_fit[0]*(
      nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))           

 
    # Get the left and right lane line pixel locations  
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]  
 
   
     
    # Fit a second order polynomial curve to each lane line
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

         
    # Create the x and y values to plot on the image
    ploty = np.linspace(0, warped_frame.shape[0]-1, warped_frame.shape[0]) 
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Generate images to draw on
    out_img = np.dstack((warped_frame,warped_frame, (warped_frame)))*255
    window_img = np.zeros_like(out_img)
             
    # Add color to the left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
             
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255,0), lineType=cv2.LINE_8)
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255,0), lineType=cv2.LINE_8)
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return result,left_fitx,right_fitx,ploty,righty,lefty,rightx,leftx

## CREATE FINAL IMAGE ## 
def overlay(orig_frame,warped_frame,left_fitx,right_fitx,ploty,Minv):

    # Generate an image to draw the lane lines on 
    warp_zero = np.zeros_like(warped_frame).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))       
         
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx,ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx,ploty])))])
    pts = np.hstack((pts_left, pts_right))
         
    # Draw lane on the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
 
    # Warp the blank back to original image space using inverse perspective 
    # matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (orig_frame.shape[1],orig_frame.shape[0]))
     
    # Combine the result with the original image
    result = cv2.addWeighted(orig_frame, 1, newwarp, 0.3, 0)
    return result

## CALCULATE R AND L CURVATURE 
def calculate_curvature(ploty,righty,lefty,rightx,leftx,YM_PER_PIX = 10.0 / 1000,XM_PER_PIX = 3.7 / 781):

    # Set the y-value where we want to calculate the road curvature.
    # Select the maximum y-value, which is the bottom of the frame.
    y_eval = np.max(ploty)    
 
    # Fit polynomial curves to the real world environment
    left_fit_cr = np.polyfit(lefty * YM_PER_PIX, leftx * (XM_PER_PIX), 2)
    right_fit_cr = np.polyfit(righty * YM_PER_PIX, rightx * (XM_PER_PIX), 2)
             
    # Calculate the radii of curvature
    left_curvem = ((1 + (2*left_fit_cr[0]*y_eval*YM_PER_PIX + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curvem = ((1 + (2*right_fit_cr[0]*y_eval*YM_PER_PIX + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
             
    left_curvem = left_curvem
    right_curvem = right_curvem
 
    return left_curvem, right_curvem

## CALCULATE CAR OFFSET FROM THE MID ##
def calculate_car_position(orig_frame,left_fit,right_fit,XM_PER_PIX = 3.7 / 781):
    # Assume the camera is centered in the image.
    # Get position of car in centimeters
    car_location = orig_frame.shape[1] / 2
 
    # Fine the x coordinate of the lane line bottom
    height = orig_frame.shape[0]
    bottom_left = left_fit[0]*height**2 + left_fit[1]*height + left_fit[2]
    bottom_right = right_fit[0]*height**2 + right_fit[1]*height + right_fit[2]
 
    center_lane = (bottom_right - bottom_left)/2 + bottom_left 
    center_offset = (np.abs(car_location) - np.abs(center_lane)) * XM_PER_PIX * 100
   
    return center_offset


def debugConsole(frame,preprocess_image,roi_image,warp_image,frame_sliding,result,final_image,left_curve,right_curve,center_offset,left_fitx,right_fitx,ploty,width,height,debugON=False):

    if debugON==True:
        cv2.imshow("Original Image",frame)
        cv2.imshow("After process Image(Threshold and Canny)",preprocess_image)
        cv2.imshow("ROI Image",roi_image)
        cv2.imshow("Image after Perspective Transformation",warp_image)
        cv2.putText(final_image,'Curve Radius: '+str((left_curve+right_curve)/2)[:7]+' m', (int((5/600)*width), int((20/338)*height)), cv2.FONT_HERSHEY_SIMPLEX, (float((0.5/600)*width)),(255,255,255),2,cv2.LINE_AA)
        cv2.putText(final_image,'Center Offset: '+str(center_offset)[:7]+' cm', (int((5/600)*width), int((40/338)*height)), cv2.FONT_HERSHEY_SIMPLEX, (float((0.5/600)*width)),(255,255,255),2,cv2.LINE_AA)
        cv2.imshow("Conclusion",final_image)
        cv2.imshow("Sliding Windows",frame_sliding)

        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow',linewidth=3)
        plt.plot(right_fitx, ploty, color='yellow',linewidth=3)
        plt.pause(0.000001)
        plt.clf()



## MAIN CODE ## 
if __name__ == "__main__":
    cap = cv2.VideoCapture("C:/Users/fzlfrkkms/Desktop/SEMINAR/lanedetector/LaneDetectionVideos/videos/3.mp4")
    while cap.isOpened():
        ret,frame=cap.read()
        if ret==True:
            
            ## MAIN PREPROCESS ##
            HLS_image=processHLS(frame)
            thresholdL_image=threshold(HLS_image[:, :, 1],plot=False)
            edge_image=processEdge(thresholdL_image,plot=False)
            edge_image=processGauss(edge_image)
            thresholdS_image=threshold(HLS_image[:, :, 2],thresh=(128,255))
            thresholdR_image=threshold(frame[:, :, 2],plot=False)
            preprocess_image=binaryProcess(thresholdS_image, thresholdR_image,edge_image, plot=False)
            
            ## PERSPECTIVE TRANSFORM ## 
            roi_image,roi,desired,M,Minv=calcTM(preprocess_image,offset=180,point_offset=60,y_offset=90,toright=90,height_crop=55,plot=False)
            warp_image=perspectiveTransform(roi_image,M,plot=False)

            ## HISTOGRAM OF THE IMAGE ##
            histogram,peak_left,peak_right=lane_histogram(warp_image, height_start=800, height_end=1250,plot=False)

            ## Find lines by slicing window techniques ##
            left_poly,right_poly,frame_sliding=lane_polys(warp_image,margin=160,no_of_windows=10,minpix=80)
            result,left_fitx,right_fitx,ploty,righty,lefty,rightx,leftx=lane_line(warp_image,left_poly, right_poly,margin=160,no_of_windows=10,minpix=80)
            final_image=overlay(frame,warp_image,left_fitx,right_fitx,ploty,Minv)
            

            ## CALCULATE CAR OFFSET AND LANE CURVATURE ## 
            left_curve, right_curve=calculate_curvature(ploty,righty,lefty,rightx,leftx,YM_PER_PIX = 10.0 / 1000,XM_PER_PIX = 3.7 / 781)
            center_offset=calculate_car_position(frame,left_poly,right_poly,XM_PER_PIX = 3.7 / 781)
            
            ## DEBUG CONSOLE ## 
            debugConsole(frame,preprocess_image,roi_image,warp_image,frame_sliding,result,final_image,left_curve,right_curve,center_offset,left_fitx,right_fitx,ploty,width=frame.shape[1],height=frame.shape[0],debugON=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()     
         