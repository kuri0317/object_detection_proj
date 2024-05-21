import numpy as np
import cv2 as cv
import argparse
import constants
from collections import defaultdict
from numpy.core.multiarray import ndarray


def generalized_hough_transform(model: dict, scene_img: ndarray, threshold=constants.THRESHOLD, min_matches=constants.MIN_MATCHES):
    """
    Compute ght alghorithm using SIFT descriptors

    Parameters:
    model: dict object as the output of getModelKeypointsDescriptors()
    scene_img: path to a scene in the constants.SCENE_PATH directory
    threshold=0.75 threshold for the ratio test
    min_matches=200: minimum number of mathces that need to be found in a scene
    """

    # Compute keypoints and descriptors for the scene image
    sift = cv.SIFT_create()
    keypoints_scene, descriptors_scene = sift.detectAndCompute(scene_img,None)

    # Match descriptors between model and scene
    flann = cv.FlannBasedMatcher()
    matches = flann.knnMatch(model['descriptors'], descriptors_scene, k=2)

    # filter matches between keypoints with thresholding
    good_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)
    print(f"Number of good matches: {len(good_matches)}")

    # terminate if matches are lower then threshold
    if len(good_matches) <min_matches:
        return None, 0, None, None, None

    # init accumulator
    accumulator = defaultdict(int)
    
    for match in good_matches:

        # set indexes of the keypoint that had been matched
        model_idx = match.queryIdx
        scene_idx = match.trainIdx
        
        # get keypoint that had been matched
        model_kp = model['keypoints'][model_idx]
        scene_kp = keypoints_scene[scene_idx]
        
        model_vector = model['vectors'][model_idx]

        # compute the scale between the model and the scene 
        scale = scene_kp.size / model_kp.size
        #scale = 1
        rotated_vector = scale * model_vector
        
        scene_centroid = np.array(scene_kp.pt) - rotated_vector
        accumulator[tuple(scene_centroid)] += 1


    # Find the maximum votes in the accumulator
    max_votes = max(accumulator.values())

    w, h =  model['model_img'].shape[::-1]
    possible_centroids = [pos for pos, votes in accumulator.items() if votes == max_votes]

    return possible_centroids, max_votes, keypoints_scene, descriptors_scene,good_matches

