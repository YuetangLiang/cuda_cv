import cv2
import numpy as np

import sys


import pylcv


img_src = cv2.imread("test/resources/bgr_2880_1860_warp_ori.jpeg")
img_dest = np.zeros_like(img_src)
print(img_src.shape)

xform = np.array(
    [
        0.996600978683475,
        0.1182040112496254,
        -125.6150237041926,
        -0.006767036783381765,
        1.053820548892408,
        -63.34798599158658,
        -1.06319498835945e-06,
        5.260880922951616e-05,
        1,
    ]
)

border_val = np.array([10.0, 10.0, 10.0, 10.0])

shape = img_src.shape
warp_presp = pylcv.WarpPerspective(shape[0], shape[1], "BGR_8U")
warp_presp.Operator(
    img_src,
    img_dest,
    xform,
    0,
    0,
    border_val
)


expect_result = cv2.imread("test/resources/bgr_2880_1860_warp.bmp")
np.testing.assert_allclose(img_dest, expect_result)
