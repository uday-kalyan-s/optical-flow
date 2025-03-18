from ai2thor.controller import Controller
import cv2 as cv
import numpy as np
import lk_final as lk
from ai2thor.platform import CloudRendering
import time

start = time.time()

controller = Controller(scene="FloorPlan10")

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners = 300, qualityLevel = 0.2, minDistance = 15, blockSize = 1)

# window size is a number such that in a window the actual width of it is window_size*2+1. i want actual size odd ebcause it involves slightly diff impl
lk_params = dict(min_eigval=0.1**2, eig_val_ratio=10**3, flow_color=(0,255,0), flow_thickness=5, max_level=5, win_edge_dist=3)

e = controller.step(action="Pass")

prev_gray = cv.cvtColor(e.cv2img, cv.COLOR_BGR2GRAY)
prev_features = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)

line_mask = np.zeros_like(prev_gray)
time_delta=1

out = cv.VideoWriter('out.avi', cv.VideoWriter_fourcc(*'MJPG'), 15, prev_gray.shape)

while True:
    e = controller.step(action="Pass")
    gray = cv.cvtColor(e.cv2img, cv.COLOR_BGR2GRAY)
    out.write(gray)
    now = time.time()
    if (now - start) > 10:
        break

cv.destroyAllWindows()
out.release()