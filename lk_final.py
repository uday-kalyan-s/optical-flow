import cv2 as cv
import numpy as np

# ignore name just calculates average value of all pixels in a matrix, used in blurring
def squash(matrix):
    rl = matrix.ravel()
    return np.sum(rl)/len(rl)

# takes a matrix of dimension 3^n, 3^n where n is pyr level and blurs it to 3*3
def blur(matrix, window_size):
    subbox_s = len(matrix)//window_size
    new_mat = np.zeros((window_size,window_size))
    for i in range(window_size):
        for j in range(window_size):
            new_mat[i,j] = squash(matrix[i*subbox_s:(i+1)*subbox_s, j*subbox_s:(j+1)*subbox_s])
    return new_mat

# pyramidal implementation
def lk(features, prev_gray, cur_gray, line_mask, time_delta, flow_color, min_eigval, eig_val_ratio, flow_thickness, max_level, win_edge_dist):
    window_size = (win_edge_dist*2)+1
    max_y, max_x = prev_gray.shape
    corners = np.int64(features)
    next_features = []
    # applying lucas kanade
    for cor in corners:
        pyr_level = 1
        while pyr_level <= max_level:
            coeff_mat = np.zeros((2,2), dtype=np.int64)
            rhs_mat = np.zeros((2,1), dtype=np.int64)
            cor = cor.ravel()
            x,y = cor
            left_shift = -(window_size**pyr_level-1)//2 # how left from current pixel do u start slicing
            right_shift = (window_size**pyr_level+1)//2 # how right from current pixel do u stop slicing
            if ((x+left_shift) < 0) or ((x+right_shift) >= max_x) or ((y+left_shift) < 0) or ((y+right_shift) >= max_y):
                # goes out of bounds, no point tracking this corner
                break
            else:
                shortened_mat = cur_gray[
                    y+left_shift : y+right_shift,
                    x+left_shift : x+right_shift
                ]
                shortened_mat_prev = prev_gray[
                    y+left_shift : y+right_shift,
                    x+left_shift : x+right_shift
                ]
            blurred_mat = blur(shortened_mat, window_size)
            blurred_mat_prev = blur(shortened_mat_prev, window_size)
            for i in range(-win_edge_dist,win_edge_dist+1):
                for j in range(-win_edge_dist,win_edge_dist+1):
                    try:
                        i_x = (blurred_mat[i, j+1] - blurred_mat[i, j])/pyr_level
                    except:
                        i_x = (blurred_mat[i, j] - blurred_mat[i, j-1])/pyr_level
                    try:
                        i_y = (blurred_mat[i+1, j] - blurred_mat[i, j])/pyr_level
                    except:
                        i_y = (blurred_mat[i, j] - blurred_mat[i-1, j])/pyr_level
                    i_t = (blurred_mat[i, j] - blurred_mat_prev[i, j])/time_delta
                    coeff_mat += np.array([[i_x**2, i_x*i_y], [i_x*i_y, i_y**2]], dtype=np.int64)
                    rhs_mat += np.array([[-i_x*i_t], [-i_y*i_t]], dtype=np.int64)

            # enforce quality checks
            l1, l2 = np.linalg.eigvals(coeff_mat)
            if np.linalg.det(coeff_mat) != 0 and 1/eig_val_ratio < l1/l2 < eig_val_ratio and l1 > min_eigval and l2 > min_eigval:
                soln = (np.linalg.inv(coeff_mat)) @ rhs_mat
            else: # quality isnt good, blur and try
                pyr_level += 1
                continue

            # soln does exist
            v = soln.ravel().astype(np.int64)
            vx, vy = v*time_delta*pyr_level

            # 2nd check
            if vx == 0 and vy == 0: # if both are at same pos then dont bother drawing
                pyr_level += 1
                continue
            newp = (vx+x, vy+y)
            cv.line(line_mask, newp, cor, flow_color, flow_thickness)
            next_features.append([list(newp)])
            break
    return next_features

# adds corner circles detected by shi-tomasi
def feature_mask(frame, features, color):
    # draw the corner detection points first
    mask = np.zeros_like(frame)
    corners = np.int64(features)
    for i in corners:
        x,y = i[0] # converts [[x,y]] to [x,y]
        cv.circle(mask, (x,y), 3, color, -1)
    return mask
