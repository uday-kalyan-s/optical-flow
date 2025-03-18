import cv2 as cv
import numpy as np
import lk_final as lk

video_scenes = [[16.5, 18.5], [7,9]]
scene_no = 0
current_scene = video_scenes[scene_no]
time_delta = 5
feature_color = (255,0,0)


# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners = 300, qualityLevel = 0.2, minDistance = 15, blockSize = 1)

# window size is a number such that in a window the actual width of it is window_size*2+1. i want actual size odd ebcause it involves slightly diff impl
lk_params = dict(min_eigval=0.1**2, eig_val_ratio=10**3, flow_color=(0,255,0), flow_thickness=5, max_level=5, win_edge_dist=3)

# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("resources/video.mp4")
cap.set(cv.CAP_PROP_POS_MSEC, current_scene[0]*1000)

# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = cap.read()

# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

prev_features = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
# Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
line_mask = np.zeros_like(first_frame)

while(cap.isOpened()):
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Calculates sparse optical flow by Lucas-Kanade method

    # contains features of the prevois frame
    prev_features = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
    features_mask = lk.feature_mask(frame, prev_features, feature_color)

    # update the line_mask with lines
    prev_features = lk.lk(prev_features, prev_gray, gray, line_mask, time_delta, **lk_params)
    # prev_features = np.setxor1d(prev_features, cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params))
    # Updates previous frame
    prev_gray = gray.copy()

    # draw the frames with masks
    cv.imshow("flow", cv.add(frame, cv.add(line_mask, features_mask)))
    # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
    if cv.waitKey(time_delta) & 0xFF == ord('q') or cap.get(cv.CAP_PROP_POS_MSEC) > current_scene[1]*1000:
        break
# The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()
