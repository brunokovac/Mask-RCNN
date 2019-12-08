import tensorflow as tf
import numpy as np
from backbone import *

def get_anchor_dimensions(scales, ratios):
    dimensions = []
    for scale in scales:
        for ratio in ratios:
            wr, hr = ratio

            if wr == 1:
                dimensions.append([int(round(np.sqrt(scale/hr))), int(round(np.sqrt(scale*hr)))])
            elif hr == 1:
                dimensions.append([int(round(np.sqrt(scale * wr))), int(round(np.sqrt(scale / wr)))])
    return dimensions

if __name__ == "__main__":
    ds = get_anchor_dimensions([64**2, 128**2], [(1, 1), (1, 2), (2, 1)])
    print(ds)
    print(ds[0][0] * ds[0][1])
    print(ds[1][0] * ds[1][1])