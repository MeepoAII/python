import numpy as np
import cv2

image = cv2.imread("/home/lab-1/Downloads/meepo.jpg")
cv2.imshow("Original",image)
cv2.waitKey(0)

#R、G、B分量的提取
(B,G,R) = cv2.split(image)#提取R、G、B分量
cv2.imshow("Red",R)
cv2.imshow("Green",G)
cv2.imshow("Blue",B)
print(R)
print(G)
print(B)
cv2.waitKey(0)
