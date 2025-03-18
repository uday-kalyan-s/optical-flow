import numpy as np

def genmask(pixels, height, width, color):
    mask = np.zeros((height, width, 3))
    print(mask)
    for pt in pixels:
        h,w = pt[0]
        print(h,w)
        mask[int(h)][int(w)] = color
    return mask