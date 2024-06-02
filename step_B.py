import numpy as np
import cv2 as cv
import constants
from collections import defaultdict
from ght_sift import  generalized_hough_transform
import argparse
from bbox import *

# parameter parser function
def getParams():
    '''
    parse command line arguments

    '''
    parser = argparse.ArgumentParser(prog='object_detection_project',description='box detection project',epilog='credits carnivuth,kuri')
    parser.add_argument('-t','--threshold',default=constants.THRESHOLD,help='thrashold for ratio test',type=float)
    parser.add_argument('-m','--minMatches',default=constants.MIN_MATCHES,help='minimum number of matches for detecting the model in the target image',type=int)
    return parser.parse_args()

def display_results(scene_img, model, matches, scene_keypoints, bbox_props_list, matchesMask):

    instance_count= len(bbox_props_list)
    scene_with_instances= cv.cvtColor(scene_img, cv.COLOR_GRAY2BGR)

    for bbox in bbox_props_list:
        corners= np.int32(bbox['corners'])
        if bbox['valid_bbox']:
            cv.polylines(scene_with_instances, [corners], True, (0,255,0),2)

    final = cv.drawMatches(model._model_img, model._keypoints, scene_with_instances, scene_keypoints, matches, None, matchesMask= matchesMask)

    print (f'number of instances found with RANSAC: {instance_count}')
    cv.imshow('detect instance', final)
    cv.waitKey(0)
    cv.destroyAllWindows()

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
    models = Bbox.getModelKeypointsDescriptors(product_paths)

    # ONLINE PHASE: run object detection on the scene images with the GHT + SIFT pipeline
    for scene_path in scene_paths:

        scene_count = {}
        print(f"SEARCHING in Scene: {scene_path}")

        # read scene image
        scene_img = cv.imread(scene_path, cv.IMREAD_GRAYSCALE)

        for model in models:
            results =  generalized_hough_transform(model, scene_img, threshold, min_matches)
            if results != None:
                print(f'model :{model._model_name}')
                print(f'number of instances found :{len(results.centroids)}')
                print(f'max_votes:{results.max_score}')

                # compute scale between model image and scene image
                #scale_w = scene_img.shape[0] /model._model_img.shape[0]
                #scale_h = scene_img.shape[1]/ model._model_img.shape[1]

                #for bbox_props in bbox_props_list:
                #    if model._model_name in scene_count:
                #        scene_count[model._model_name]['n_instance'] += 1
                #    else:
                #        scene_count[model._model_name] = {'n_instance': 1}

                #display_results(scene_img, model, results.matches , results.scene_descriptors, bbox_props_list, matchesMask)
                #print(f'numero istanze trovate con box e ransac:{scene_count}')
        #counts[scene_path] = scene_count

    return counts

# main
scene_paths = [constants.SCENES_PATH+'/m1.png', constants.SCENES_PATH+'/m2.png', constants.SCENES_PATH+'/m3.png', constants.SCENES_PATH+'/m4.png', constants.SCENES_PATH+'/m5.png']
product_paths = [constants.MODELS_PATH + '/0.jpg', constants.MODELS_PATH +'/1.jpg', constants.MODELS_PATH +'/11.jpg', constants.MODELS_PATH +'/19.jpg', constants.MODELS_PATH +'/24.jpg', constants.MODELS_PATH +'/26.jpg', constants.MODELS_PATH +'/25.jpg']

args = getParams()
counts = find_instances(scene_paths, product_paths, args.threshold, args.minMatches)
