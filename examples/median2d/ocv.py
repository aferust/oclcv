# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 18:56:26 2024

@author: user
"""

import cv2
import numpy as np
import time

img = cv2.imread('lena.png')


start_time = time.time()
median = cv2.medianBlur(img, 5)
end_time = time.time()

execution_time_ms = (end_time - start_time) * 1000
print("Execution time:", execution_time_ms, "milliseconds")

cv2.imshow('median', median)
cv2.waitKey(0)
cv2.destroyAllWindows()