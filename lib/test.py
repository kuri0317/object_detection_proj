from utils import  getModelKeypointsDescriptors
import cv2 as cv
import numpy as np
product_paths = ['0.jpg', '1.jpg', '11.jpg', '19.jpg', '24.jpg', '26.jpg', '25.jpg']
models = getModelKeypointsDescriptors(product_paths)
for model in models:

    img= cv.drawKeypoints(model['model_img'],model['keypoints'],None)
    print(model['centroid'])
    print(model['vectors'][0])
    for keypoint in model['keypoints']: img= cv.line(img,model['centroid'].astype(int),np.array(keypoint.pt).astype(int),color=(255,0,0),thickness=2)
    img= cv.circle(img,model['centroid'].astype(int),radius=50,color=(0,0,255),thickness=2)

    cv.imshow('kp and center',img)
    cv.waitKey()
