import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import getModelKeypointsDescriptors
import constants

"""
Test on scene image: {e1.png, e2.png, e3.png, e4.png, e5.png}
Use product images: {0.jpg, 1.jpg, 11.jpg, 19.jpg, 24.jpg, 26.jpg, 25.jpg}


Develop an object detection system to identify single instance of products given: one reference image for
each item and a scene image. The system should be able to correctly identify all the product in the shelves
image. One way to solve this task could be the use of local invariant feature as explained in lab session 5.

"""
'''
Implement an Object Detection Pipeline in OpenCV with Local Invariant Features

4 steps:
1. Detection: identify salient and repeatable points 'keypoints' in both reference and target images;
2. Description: create a unique descriptior for each point, based on its local pixel neighborhood;
3. Matching: match points from reference and target images according to a similarity function between the descriptors;
4. Position Estimation: estimate the position of the object in the target image given the matching points. 
'''


# model images
scene_paths = ['e1.png', 'e2.png', 'e3.png', 'e4.png', 'e5.png']
product_paths = ['0.jpg', '1.jpg', '11.jpg', '19.jpg', '24.jpg', '26.jpg', '25.jpg']


# matcher BFMatcher 
min_match_threshold = 131


models = getModelKeypointsDescriptors(product_paths)

for scene_img_path in scene_paths:

    # Carica l'immagine di test
    scene_img = cv2.imread(constants.SCENES_PATH +'/'+  scene_img_path, cv2.IMREAD_GRAYSCALE)

    # keypoints e descrizioni scene_img
    sift = cv2.SIFT_create()
    keypoints_scene, descriptors_scene = sift.detectAndCompute(scene_img, None)
    
    print(f'computing scene {scene_img_path}')
    for model in models:
        
        # matching delle descrizioni keypoints tra img
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(model["descriptors"], descriptors_scene, k=2)

        # Filtra i match basati sulla distanza tra i due match più vicini (ratio test)
        good_matches = []
        for m, n in matches:
            if m.distance < constants.GOOD_MATCH_THRESHOLD * n.distance:
                good_matches.append(m)
        
        #img_matches = cv2.drawMatches(model["model_img"], model["keypoints"], scene_img, keypoints_scene, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #cv2.imshow('Matches', img_matches)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # numero di match trovati per coppia 
        

        if len(good_matches) >= min_match_threshold:
            print(f"instance {model['model_name']} found")
        else:
            print(f"instance {model['model_name']} not found")



