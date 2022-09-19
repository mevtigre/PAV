#region IMPORT
# %%
## IMPORT PART
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
#endregion

#region COLOR SPACE CONVERSION
# %%
## CONVERT HLS ## 
def processHLS(frame,plot=False):
    hls_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    if plot== True:
        cv2.imshow("Hls image",hls_image)
    return hls_image
#endregion

#region PREPROCESS
# %%
## WHITE AND YELLOW THRESHOLD ## 
def color_thresholding(frame,plot=False):

    hls = processHLS(frame,plot=False)
    # white threshold
    white_threshold = np.zeros_like(hls[:,:,0])
    white_threshold[((hls[:,:,0] >= 0) & (hls[:,:,0] <= 255)) & ((hls[:,:,1] >= 200) & (hls[:,:,1] <= 255)) & ((hls[:,:,2] >= 0) & (hls[:,:,2] <= 255))] = 255
    #  yellow threshold
    yellow_threshold = np.zeros_like(hls[:,:,0])
    yellow_threshold[((hls[:,:,0] >= 15) & (hls[:,:,0] <= 55)) & ((hls[:,:,1] >= 30) & (hls[:,:,1] <= 204)) & ((hls[:,:,2] >= 115) & (hls[:,:,2] <= 255))] = 255
    #yellow_threshold=np.array(yellow_threshold, dtype = np.uint8 )
    # Combination
    color_thresholded = np.zeros_like(hls[:,:,0])
    color_thresholded[(yellow_threshold == 255) | (white_threshold == 255)] = 255
    if plot== True:
      cv2.imshow("Hls w and y extracted",color_thresholded)
    return color_thresholded

## Apply Sobel Operator ##
def sobel_abs(gray_img, dir=1, kernel_size=3, thres=(0, 255),plot=False):

    if dir == 0: # in the direction of y
        sobel = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size) 
    else: # in the direction of x
        sobel = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size) 

    scaled = np.uint8(255 * sobel / np.max(np.absolute(sobel)))
    
    gradient = np.zeros_like(scaled)
    gradient[(thres[0] <= scaled) & (scaled <= thres[1])] = 255
    if plot== True:
      cv2.imshow("Sobel image",gradient)
    return gradient

def sobel_magnitude(gray_img, kernel_size=3, thres=(0, 255),plot=False):

    gradient_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size)
    gradient_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size)
    
    gradient_xy = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    scaled_gradient = np.uint8(255 * gradient_xy / np.max(gradient_xy))
    
    binary_gradient = np.zeros_like(scaled_gradient)
    binary_gradient[(scaled_gradient <= thres[1]) & (scaled_gradient >= thres[0])] = 255
    if plot== True:
      cv2.imshow("Sobel magnitude image",binary_gradient)
    return binary_gradient

def sobel_direction(gray_img, kernel_size=3, thres=(0, np.pi/2)):

    abs_gradient_y = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size))
    abs_gradient_x = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size))
 
    gradient_angle = np.arctan2(abs_gradient_x, abs_gradient_y)

    output = np.zeros_like(gradient_angle)
    output[(gradient_angle <= thres[1]) & (gradient_angle >= thres[0])] = 255
    
    return output

def combined_sobels(gradientbin_x, gradientbin_y, gradientmag_xy, gray_img, kernel_size=3, angle_thres=(0, np.pi/2),plot=False):
    gradientdir_xy = sobel_direction(gray_img, kernel_size=kernel_size, thres=angle_thres)
    
    combined = np.zeros_like(gradientdir_xy)
    # Combine all sobels outputs 
    combined[(gradientbin_x == 255) | ((gradientbin_y == 255) & (gradientdir_xy == 255)) & (gradientmag_xy == 255)] = 255
    if plot== True:
      cv2.imshow("Combined sobel image",combined)  
    return combined

def binaryProcess(sobelImage, wythresImage,plot=False):
    combined_color = (np.dstack(( np.zeros_like(wythresImage), sobelImage, wythresImage)) )
    combined_gray = np.zeros_like(wythresImage)
    combined_gray[(sobelImage == 255) | (wythresImage == 255)] = 255
    if plot==True:
      cv2.imshow("Combined preprocess color image",combined_color)
      cv2.imshow("Combined preprocess color image",combined_gray)
    return combined_color,combined_gray 
#endregion

#region PERSPECTIVE TRANSFROMATION
# %%
def calcTM(test_frame,offset=300,plot=False):
  copy_combined = np.copy(test_frame)
  img_size = (test_frame.shape[1], test_frame.shape[0])
  vertices =  (190, 720), (596, 447), (685, 447), (1125, 720)
  vertices = np.array([vertices],np.int32)
  src_pts = np.float32([(190, 720), (596, 447), (685, 447), (1125, 720)])
    # Destination points are to be parallel, taken into account the image size
  dst_pts = np.float32([[offset, img_size[1]], [offset, 0],[img_size[0]-offset, 0],[img_size[0]-offset, img_size[1]] ])
  M = cv2.getPerspectiveTransform(src_pts, dst_pts)
  Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
  mask = np.zeros_like(test_frame)
  mask=cv2.fillPoly(mask,vertices,255)
  roi_image = cv2.bitwise_and(test_frame,mask)
  if plot == True:
      cv2.imshow("Frame After preprocessing and ROI", roi_image)
  return roi_image,src_pts,dst_pts,M,Minv

   
def perspectiveTransform(preprocess_image,M,plot=False):
    frame_size = (preprocess_image.shape[1], preprocess_image.shape[0])
    warp_image = cv2.warpPerspective(preprocess_image, M, frame_size, flags=cv2.INTER_LINEAR)
    if plot==True:
        cv2.imshow("Warped image",warp_image)
    return warp_image
#endregion

#region FIND LANES(SLIDING WINDOW)
# %%
def lane_histogram(img, height_start=800, height_end=1250,plot=False):
    histogram = np.sum(img[int(height_start):int(height_end),:], axis=0)
    mid = int(histogram.shape[0]/2)
    peak_right = np.argmax(histogram[mid:]) + mid # this mid offset is imp not miss
    peak_left = np.argmax(histogram[:mid])

    if plot==True:
      plt.plot(histogram)
      plt.title("Histogram Peaks")
      plt.pause(0.0000001)
      plt.clf()
      #plt.show()
    return histogram,peak_left,peak_right

def find_lane_pixels_using_histogram(warped_img,nwindows=10,margin=100,pixel_threshold=50):


    _,left_base,right_base=lane_histogram(warped_img, height_start=warped_img.shape[0]//2, height_end=1250,plot=False)

    crnt_right = right_base
    crnt_left = left_base


    nonzero_px = warped_img.nonzero()
    nonzero_pxy = np.array(nonzero_px[0])
    nonzero_pxx = np.array(nonzero_px[1])

    height = int(warped_img.shape[0]//nwindows)

    left_inds = []
    right_inds = []

    for window in range(nwindows):
        
        #define 4 edges of the window for each window
        xr_low = crnt_right - margin
        xr_high = crnt_right + margin
        xl_low = crnt_left - margin
        xl_high = crnt_left + margin

        y_low = warped_img.shape[0] - (window+1)*height
        y_high = warped_img.shape[0] - window*height

        left_pts=((nonzero_pxy >= y_low) & (nonzero_pxy < y_high) &  (nonzero_pxx >= xl_low) &  (nonzero_pxx < xl_high)).nonzero()[0]
        right_pts=((nonzero_pxy >= y_low) & (nonzero_pxy < y_high) & (nonzero_pxx >= xr_low) &  (nonzero_pxx < xr_high)).nonzero()[0]
        
        # find nonzero pixels inside the each window 
        left_inds.append(left_pts)
        right_inds.append(right_pts)
        
        if len(left_pts) > pixel_threshold:
            crnt_left = int(np.mean(nonzero_pxx[left_pts]))
        if len(right_pts) > pixel_threshold:        
            crnt_right = int(np.mean(nonzero_pxx[right_pts]))
    try:
        left_inds = np.concatenate(left_inds)
        right_inds = np.concatenate(right_inds)
    except ValueError:
        pass

    # Find a x and y position of each side of the lane, left and right
    rightx = nonzero_pxx[right_inds]
    righty = nonzero_pxy[right_inds]
    leftx = nonzero_pxx[left_inds]
    lefty = nonzero_pxy[left_inds] 


    return leftx, lefty, rightx, righty

def fit_poly(binary_warped,leftx, lefty, rightx, righty):

    try:
        params_left = np.polyfit(lefty, leftx, 2)
        params_right = np.polyfit(righty, rightx, 2)   
    except TypeError:
        params_left = [0.00001,0.00001,0.00001]
        params_right = [0.00001,0.00001,0.00001]
        
    # ploty for the visualize
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        right_poly =params_right[2] + params_right[1]*ploty + params_right[0]*ploty**2 
        left_poly = params_left[2] +  params_left[1]*ploty + params_left[0]*ploty**2 
    except TypeError:
        right_poly = 1*ploty**2 + 1*ploty
        left_poly = 1*ploty**2 + 1*ploty

    return  ploty,params_left, params_right, left_poly, right_poly
#endregion

#region CALCULATION
# %%
def measure_curvature_meters(binary_warped, left_fitx, right_fitx, ploty,ym_per_pix=30/720,xm_per_pix=3.7/700):
    
    y_eval = np.max(ploty) # find y value that we want to calculate radius on it
    right_params = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    left_params = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    
    # Left and right curvature calculation in radius
    right_curverad = ((1 + (2*right_params[0]*y_eval*ym_per_pix + right_params[1])**2)**1.5) / np.absolute(2*right_params[0])
    left_curverad = ((1 + (2*left_params[0]*y_eval*ym_per_pix + left_params[1])**2)**1.5) / np.absolute(2*left_params[0])
  
    return left_curverad, right_curverad

def measure_position_meters(binary_warped, left_param, right_param,ym_per_pix=30/720,xm_per_pix=3.7/700):


    y_max = binary_warped.shape[0]
    left_x_pos = left_param[2] + left_param[1]*y_max +  left_param[0]*y_max**2
    right_x_pos = right_param[2] +  right_param[1]*y_max + right_param[0]*y_max**2
    # calculate center
    center = (left_x_pos + right_x_pos)//2
    # if positive then on the right if negative= on the left
    pos_offset = ((binary_warped.shape[1]//2) - center) * xm_per_pix 
    return pos_offset
#endregion

#region VISUALISATION
# %%
def draw_poly_lines(binary_warped, left_fitx, right_fitx, ploty,margin=100,plot=False):     
    
    zeros = np.dstack((binary_warped, binary_warped, binary_warped))
    color_img = np.zeros_like(zeros)


    rl_w1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    rl_w2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    rline_pts = np.hstack((rl_w1, rl_w2))    
    ll_w1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    ll_w2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    lline_pts = np.hstack((ll_w1, ll_w2))


    # Drawing part
    cv2.fillPoly(color_img, np.int_([rline_pts]), (0, 255, 255))
    cv2.fillPoly(color_img, np.int_([lline_pts]), (0, 255, 255))

    final = cv2.addWeighted(zeros, 1, color_img, 0.5, 0)
    
    for i in range(ploty.shape[0]):
        cv2.circle(final, [int(left_fitx[i]),int(ploty[i])], radius=0, color=(0, 0, 255), thickness=15)
        cv2.circle(final, [int(right_fitx[i]),int(ploty[i])], radius=0, color=(255, 0, 0), thickness=15)
    if plot==True:
        cv2.imshow("Lane Debug",final)
    return final

def project_lane_info(img, binary_warped, ploty, left_fitx, right_fitx, M_inv):
    
    # Illustration part and image for it 
    zeros = np.zeros_like(binary_warped).astype(np.uint8)
    color_img = np.dstack((zeros, zeros, zeros))
    
    # Fill the lane with green 
    left_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    all_pts = np.hstack((left_pts, right_pts))
    cv2.fillPoly(color_img, np.int_([all_pts]), (0,255, 0))
    

    new_img = cv2.warpPerspective(color_img, M_inv, (img.shape[1], img.shape[0]))
    
    # Combine the result with the original image
    final_img = cv2.addWeighted(img, 1, new_img, 0.3, 0)
    
    return final_img

def debugConsole(preprocess_image,roi_image,frame_sliding,final_image,left_curve,right_curve,center_offset,left_fitx,right_fitx,ploty,width,height,debugON=False):


    resize_roi = cv2.resize(np.dstack((roi_image, roi_image, roi_image)),(192,154))#Resize the image to fit the picture in the video
    resize_preprocess = cv2.resize(np.dstack((preprocess_image, preprocess_image, preprocess_image)),(192,154))#Resize the image to fit the picture in the video
    resize_sliding_image = cv2.resize(frame_sliding,(192,154))
    cv2.putText(final_image,'Curve Radius: '+str((left_curve+right_curve)/2)[:7]+' m', (int((350/600)*width), int((20/338)*height)),cv2.FONT_HERSHEY_SIMPLEX, (float((0.5/600)*width)),(255,255,255),1,cv2.LINE_AA)
    cv2.putText(final_image,'Center Offset: '+str(center_offset)[:7]+' m', (int((350/600)*width), int((35/338)*height)),cv2.FONT_HERSHEY_SIMPLEX,(float((0.5/600)*width)),(255,255,255),1,cv2.LINE_AA)
    final_image[25:25+resize_preprocess.shape[0],25:25+resize_preprocess.shape[1]] = resize_preprocess
    final_image[25:25+resize_roi.shape[0],250:250+resize_roi.shape[1]] =  resize_roi
    final_image[25:25+resize_sliding_image.shape[0],475:475+resize_sliding_image.shape[1]] = resize_sliding_image
    if debugON==True:
        cv2.imshow("Conclusion",final_image)
    return final_image
#endregion

#region MAIN CODE PIPELINE 
# %%
## MAIN CODE ## 
if __name__ == "__main__":
    cap = cv2.VideoCapture("C:/Users/fzlfrkkms/Desktop/SEMINAR/lanedetector/LaneDetectionVideos/videos/3.mp4")
    out = cv2.VideoWriter('C:/Users/fzlfrkkms/Desktop/SEMINAR/lanedetector/LaneDetectionVideos/videos/deneme.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1280,720))
    #save = 'C:/Users/fzlfrkkms/Desktop/SEMINAR/lanedetector/LaneDetectionVideos/frames/advance/harder/'
    count=0
    while cap.isOpened():
        ret,frame=cap.read()
        if ret==True:
            start = time.time()
            ## Yellow and white threshold on Ligtness (HLS)
            wythresImage=color_thresholding(frame,plot=False)

            ## Gradient threshold using Sobel on Lightness (HLS)
            LigtImage = processHLS(frame,plot=False)[:,:,1]
            xdirSobel=sobel_abs(LigtImage,dir=1,kernel_size=15, thres=(20, 120),plot=False)
            ydirSobel=sobel_abs(LigtImage,dir=0,kernel_size=15, thres=(20, 120),plot=False)
            xydirSobel = sobel_magnitude(LigtImage, kernel_size=15, thres=(80, 200),plot=False)
            sobelImage = combined_sobels(xdirSobel, ydirSobel, xydirSobel, LigtImage, kernel_size=15, angle_thres=(np.pi/4, np.pi/2),plot=False)
            ## Combine preprocess image
            combined_color,combined_gray=binaryProcess(sobelImage, wythresImage,plot=False)
            ## Perspective Transformation and ROI
            roi_image,src_pts,dst_pts,M,Minv=calcTM(combined_gray,offset=300,plot=True)
            warp_image=perspectiveTransform(combined_gray,M,plot=False)


            ## Find lines by slicing window techniques ##
            leftx, lefty, rightx, righty=find_lane_pixels_using_histogram(warp_image,nwindows=10,margin=100,pixel_threshold=50)
            ploty,left_fit, right_fit, left_fitx, right_fitx=fit_poly(warp_image,leftx, lefty, rightx, righty)
            result_lane=draw_poly_lines(warp_image, left_fitx, right_fitx, ploty,margin=100,plot=False)


            ## Calculate curv and offset
            left_curverad, right_curverad=measure_curvature_meters(warp_image, left_fitx, right_fitx, ploty,ym_per_pix=30/720,xm_per_pix=3.7/700)
            veh_offset = measure_position_meters(warp_image, left_fit, right_fit,ym_per_pix=30/720,xm_per_pix=3.7/700)

            ## Fınal ımage
            final_image=project_lane_info(frame, warp_image, ploty, left_fitx, right_fitx, Minv)
            end = time.time()
            seconds = end - start
            print ("Time taken : {0} seconds".format(seconds))
            fps  = 1 / seconds
            print("Estimated frames per second : {0}".format(fps))

            ## DEBUG CONSOLE ## 
            debugImage=debugConsole(combined_gray,roi_image,result_lane,final_image,left_curverad,right_curverad,veh_offset,left_fitx,right_fitx,ploty,width=frame.shape[1],height=frame.shape[0],debugON=True)
            #cv2.imwrite(save+str(count)+".png", debugImage)
            #count=count+1
            out.write(debugImage)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("Number of total count : {0}".format(count))
    cap.release()
    out.release()
    cv2.destroyAllWindows()  
#endregion