import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt

import glob
from scipy.signal import find_peaks
import time


def adjust_brightness_contrast(frame, brightness=20, contrast=170):
    # Apply brightness and contrast adjustment
    adjusted_frame = cv2.convertScaleAbs(frame, alpha=contrast/127.0, beta=brightness)
  #  adjusted_frame = frame
    return adjusted_frame

def resize(img):
    return cv2.resize(img, (1136,639))

def thresholding_img(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h_channel = hls[ :, :, 0 ]
    l_channel = hls[ :, :, 1 ]
    s_channel = hls[ :, :, 2 ]

    _, sxbinary = cv2.threshold(l_channel, 150, 255, cv2.THRESH_BINARY)
    sxbinary = cv2.GaussianBlur(sxbinary, (3, 3), 0)

    # 2. Sobel edge detection on the L channel
    # l_channel = hls[:, :, 1]
    sobelx = cv2.Sobel(sxbinary, cv2.CV_64F, 1, 0, 3)
    sobelx = np.absolute(sobelx)
    sobely = cv2.Sobel(sxbinary, cv2.CV_64F, 0, 1, 3)
    sobely = np.absolute(sobely)
    sobel_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    scaled_sobel = np.uint8(255 * sobel_mag / np.max(sobel_mag))
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[ (scaled_sobel >= 110) & (scaled_sobel <= 255) ] = 1

    # 3. Threshold on the S channel
    s_channel = hls[ :, :, 2 ]
    _, s_binary = cv2.threshold(s_channel, 80, 255, cv2.THRESH_BINARY)
    # s_binary = np.zeros_like(s_channel)
    # s_binary[(s_channel >= 170) & (s_channel <= 255)] = 1

    # 4. Threshold on the R channel
    r_channel = img[ :, :, 2 ]
    _, r_thresh = cv2.threshold(r_channel, 120, 255, cv2.THRESH_BINARY)
    # r_binary = np.zeros_like(r_channel)
    # r_binary[(r_channel >= 200) & (r_channel <= 255)] = 1
    rs_binary = cv2.bitwise_or(s_binary, r_thresh)

    combined_binary = cv2.bitwise_or(rs_binary, sxbinary.astype(np.uint8))

    return combined_binary



# Define source (src) and destination (dst) points


def perspective_transform(img):

    src = np.float32([
        [ 380, 450 ],
        [ 750, 450 ],
        [ 50, 639 ],
        [ 1136, 639 ] ])

    dst = np.float32([
        [ 0, 0 ],
        [ img.shape[ 1 ], 0 ],
        [ 0, img.shape[ 0 ] ],
        [ img.shape[ 1 ], img.shape[ 0 ] ] ])

    img_size = (img.shape[ 1 ], img.shape[ 0 ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped_img = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped_img, M, Minv



def classify_lane(max_continuous_counts):
    if max_continuous_counts > 12:
        return 'Solid'
    elif max_continuous_counts >= 3 and max_continuous_counts < 6:
        return 'Bicycle'
    elif max_continuous_counts >= 7 and max_continuous_counts <= 12:
        return 'Vehicle'
    else:
        return 'Unknown'

def get_lane_color(lane_type):
    if lane_type == 'Solid':
        return [255, 255, 255]  # white
    elif lane_type == 'Bicycle':
        return [0, 255, 0]  # green
    elif lane_type == 'Vehicle':
        return [255, 50, 0]  # blue
    else:
        return [255, 255, 255]  # white for unknown

def sliding_window_search(binary_warped):
    swindow = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    midpoint = int(histogram.shape[0]//2)
    quarter_point = int(midpoint//2)
    leftx_base = np.argmax(histogram[:580])
    midx_base = np.argmax(histogram[580:870]) + 580
    rightx_base = np.argmax(histogram[875:]) + 875

    nwindows = 27
    margin = 40
    minpix = 100
    window_height = int(binary_warped.shape[0]//nwindows)

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    midx_current = midx_base
    rightx_current = rightx_base

    left_lane_inds = []
    mid_lane_inds = []
    right_lane_inds = []

    left_continuous_counts = 0
    mid_continuous_counts = 0
    right_continuous_counts = 0

    max_left_continuous_counts = 0
    max_mid_continuous_counts = 0
    max_right_continuous_counts = 0

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xmid_low = midx_current - margin
        win_xmid_high = midx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(swindow, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(swindow, (win_xmid_low, win_y_low), (win_xmid_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(swindow,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_mid_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_xmid_low) & (nonzerox < win_xmid_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
            left_continuous_counts += 1
        else:
            left_continuous_counts = 0
        if len(good_mid_inds) > minpix:
            midx_current = int(np.mean(nonzerox[good_mid_inds]))
            mid_continuous_counts += 1
        else:
            mid_continuous_counts = 0
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
            right_continuous_counts += 1
        else:
            right_continuous_counts = 0

        max_left_continuous_counts = max(max_left_continuous_counts, left_continuous_counts)
        max_mid_continuous_counts = max(max_mid_continuous_counts, mid_continuous_counts)
        max_right_continuous_counts = max(max_right_continuous_counts, right_continuous_counts)

        left_lane_inds.append(good_left_inds)
        mid_lane_inds.append(good_mid_inds)
        right_lane_inds.append(good_right_inds)

    left_lane_inds = np.concatenate(left_lane_inds)
    mid_lane_inds = np.concatenate(mid_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Determine lane type based on max_continuous_counts
    left_lane_type = classify_lane(max_left_continuous_counts)
    mid_lane_type = classify_lane(max_mid_continuous_counts)
    right_lane_type = classify_lane(max_right_continuous_counts)

    return left_lane_inds, mid_lane_inds, right_lane_inds, left_lane_type, mid_lane_type, right_lane_type, nonzerox, nonzeroy, swindow



def overlay_lanes(binary_warped, Minv, input_img, left_lane_inds, mid_lane_inds, right_lane_inds, left_lane_type, mid_lane_type, right_lane_type, nonzerox, nonzeroy):
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero)) * 255

    color = get_lane_color(left_lane_type)
    color_warp[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = color

    color = get_lane_color(mid_lane_type)
    color_warp[nonzeroy[mid_lane_inds], nonzerox[mid_lane_inds]] = color

    color = get_lane_color(right_lane_type)
    color_warp[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = color

    newwarp = cv2.warpPerspective(color_warp, Minv, (input_img.shape[1], input_img.shape[0]))
    result = cv2.addWeighted(input_img, 0.5, newwarp, 1, 0)

    return result


def detect_lanes(input_img):
    img1 = resize(input_img)
    #resized = input_img
    resized = adjust_brightness_contrast(img1)
    thresh_img = thresholding_img(resized)
    #cv2.imshow("qqq", thresh_img)
    warped_img, M, Minv = perspective_transform(thresh_img)
    #lane_inds_list, nonzerox, nonzeroy, lane_type_list, swindow, histogram = sliding_window_search(warped_img)
    left_lane_inds, mid_lane_inds, right_lane_inds, left_lane_type, mid_lane_type, right_lane_type, nonzerox, nonzeroy, swindow = sliding_window_search(warped_img)
    #left_curverad, right_curverad, vehicle_position = calculate_curvature(warped_img, lane_inds_list, nonzerox, nonzeroy)
   # result = overlay_lanes(warped_img, Minv, resized, lane_inds_list, nonzerox, nonzeroy, lane_type_list)
    result = overlay_lanes(warped_img, Minv, resized, left_lane_inds, mid_lane_inds, right_lane_inds, left_lane_type, mid_lane_type, right_lane_type, nonzerox, nonzeroy)
    return result, swindow, left_lane_type, mid_lane_type, right_lane_type
