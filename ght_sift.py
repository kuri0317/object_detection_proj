import numpy as np
import cv2 as cv
import argparse
import constants
import dataclasses
from collections import defaultdict
from numpy.core.multiarray import ndarray
from model import Model

def generalized_hough_transform(model: Model, scene_img: ndarray, threshold=constants.THRESHOLD, min_matches=constants.MIN_MATCHES):
    """
    Compute ght alghorithm using SIFT descriptors

    Parameters:
    -----------
        model: Model
            dict object as the output of getModelKeypointsDescriptors()
        scene_img: nd.array
            image of a scene where to looking for mathces
        threshold: float default=0.75
            threshold for the ratio test
        min_matches: int default=200
            minimum number of mathces that need to be found in a scene
    Returns
    -----------
    results: GhtOutput
        dataclass with output information of the ght process
        returns None when no sufficient matches are found

    """

    results = GhtOutput(scene_img)

    # Compute keypoints and descriptors for the scene image
    sift = cv.SIFT_create()
    results.scene_keypoints, results.scene_descriptors = sift.detectAndCompute(scene_img,None)

    # Match descriptors between model and scene
    flann = cv.FlannBasedMatcher()
    matches = flann.knnMatch(model._descriptors, results.scene_descriptors, k=2)

    # filter matches between keypoints with thresholding
    for m, n in matches:
        if m.distance < threshold * n.distance:
            results.matches.append(m)

    # terminate if matches are lower then threshold
    if len(results.matches) <min_matches:
        return None

    # init accumulator
    accumulator = defaultdict(int)

    # voting process
    for match in results.matches:

        # set indexes of the keypoint that had been matched
        model_idx = match.queryIdx
        scene_idx = match.trainIdx

        # get keypoint that had been matched
        model_kp = model._keypoints[model_idx]
        scene_kp = results.scene_keypoints[scene_idx]
        model_vector = model._vectors[model_idx]

        # compute the scale between the model and the scene
        scale = scene_kp.size / model_kp.size
        scaled_vector = scale * model_vector

        scene_centroid = np.array(scene_kp.pt) - scaled_vector
        accumulator[tuple(scene_centroid)] += 1

    # Find the maximum votes in the accumulator
    results.max_score= max(accumulator.values())

    results.centroids= [pos for pos, votes in accumulator.items() if votes == results.max_score]

    return results

@dataclasses.dataclass
class GhtOutput:
    centroids: list[np.ndarray]
    scene: np.ndarray
    max_score: int
    scene_keypoints: np.ndarray
    scene_descriptors: np.ndarray
    matches: list

    def __init__(self,scene: np.ndarray):
        self.scene= scene
        self.matches = []
