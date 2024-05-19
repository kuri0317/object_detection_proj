import numpy as np
import cv2 as cv
import constants
from collections import defaultdict
from utils import  getModelKeypointsDescriptors
from ght_sift import  generalized_hough_transform
import argparse

# parameter parser function
def getParams():
    parser = argparse.ArgumentParser(prog='object_detection_project',description='box detection project',epilog='credits carnivuth,kuri')
    parser.add_argument('-t','--threshold',default='0.5',help='thrashold for ratio test',type=float)
    parser.add_argument('-m','--minMatches',default='200',help='minimum number of matches for detecting the model in the target image',type=int)
    return parser.parse_args()

def find_instances(scene_paths, product_paths, threshold=constants.THRESHOLD, min_matches=constants.MIN_MATCHES):
    """
    find instances of the model images in the scene images using GHT and SIFT descriptors

    Parameters:
    scene_paths: image names in the constants.SCENES_PATH directory
    product_paths: image names in the constants.MODELS_PATH directory 
    threshold=0.75 threshold for the ratio test
    min_matches=200: minimum number of mathces that need to be found in a scene
    """

    counts = {}

    # OFFLINE PHASE: compute keypoint,descriptors,vectors of the model images
    models = getModelKeypointsDescriptors(product_paths)
    
    # ONLINE PHASE: run object detection on the scene images with the GHT + SIFT pipeline
    for scene_path in scene_paths:

        scene_count = {}

        # read scene image
        scene_img = cv.imread(constants.SCENES_PATH + '/' + scene_path, cv.IMREAD_GRAYSCALE)

        for model in models:
            
            # compute scale between model image and scene image
            scale_w = scene_img.shape[0] /model['model_img'].shape[0]
            scale_h = scene_img.shape[1]/ model['model_img'].shape[1]

            # run the GHT + SIFT on the scene image
            possible_centroids, max_votes, scene_keypoints, scene_descriptors,matches= generalized_hough_transform(model, scene_img, threshold,min_matches)

            if  possible_centroids != None :

                scene_with_instances = cv.cvtColor(scene_img, cv.COLOR_GRAY2BGR)

                # draw bounding box for each possible centroid
                for centroid_pos in possible_centroids:
                    center_x, center_y = centroid_pos
                    w, h =  model['model_img'].shape[::-1]
                    w = w * scale_w
                    h = h * scale_h
                    starting_point= (int(center_x-(w/2)),int(center_y-(h/2)))
                    ending_point= (int(center_x+(w/2)),int(center_y+(h/2)))
                    cv.rectangle(scene_with_instances,starting_point,ending_point,(0, 255, 0), 2)
                    final_image=cv.drawMatches(model['model_img'],model['keypoints'],scene_with_instances,scene_keypoints,matches,None)
                
                cv.imshow('Scene with instances', final_image)
                cv.waitKey(0)
                cv.destroyAllWindows()

                # update count of instances
                if model['model_name'] in scene_count:
                    scene_count[model['model_name']]['n_instance'] += 1
                else:
                    scene_count[model['model_name']] = {
                        'n_instance': 1,
                        'centroid': model['centroid'],
                        'vectors': model['vectors']
                    }


                print(f"numero istanze trovate {scene_count}")
        counts[scene_path] = scene_count

    return counts

# main
scene_paths = ['m1.png', 'm2.png', 'm3.png', 'm4.png', 'm5.png']
product_paths = ['0.jpg', '1.jpg', '11.jpg', '19.jpg', '24.jpg', '26.jpg', '25.jpg']

args = getParams()
counts = find_instances(scene_paths, product_paths, args.threshold, args.minMatches)

