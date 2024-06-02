from model import Model
from ght_sift import generalized_hough_transform
import cv2 as cv
scene = cv.imread("object_detection_project/scenes/m1.png",cv.IMREAD_GRAYSCALE)
a = Model("object_detection_project/models/0.jpg")
p,i,c,d,e=generalized_hough_transform(a,scene)
