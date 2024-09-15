import numpy as np
import cv2 as cv
import lib.constants as constants
import dataclasses
import lib.accumulator as accumulator
from lib.bbox import Bbox
from lib.model import Model

@dataclasses.dataclass
class ModelFound:
    '''
    Output class for find instances process
    '''
    model: Model
    n_instances: int
    centroids: list[np.ndarray]
    matches: list
    bboxes: list[Bbox]

    def __init__(self,model: Model,n_instances: int, matches: list , centroids: list ):
        self.model= model
        self.n_instances= n_instances
        self.matches= matches
        self.centroids= centroids

@dataclasses.dataclass
class SceneAnalisys:
    '''
    Output class for find instances process
    '''

    scene_name: str
    scene: np.ndarray
    model_instances: list[ModelFound]

    def __init__(self,scene_name:str,scene: np.ndarray):
        self.scene_name= scene_name
        self.scene= scene
        self.model_instances=[]

@dataclasses.dataclass
class GhtOutput:
    '''
    Output class for ght process
    '''

    centroids: list[np.ndarray]
    scene: np.ndarray
    max_score: int
    scene_keypoints: np.ndarray
    scene_descriptors: np.ndarray
    matches: list

    def __init__(self,scene: np.ndarray):
        self.scene= scene
        self.matches = []

def generalized_hough_transform(model: Model, scene_img: np.ndarray, threshold=constants.THRESHOLD, min_matches=constants.MIN_MATCHES,cell_size=constants.CELL_SIZE):
    """
    Compute ght alghorithm using SIFT descriptors

    Parameters:
    -----------
        model: Model
            dict object as the output of getModelKeypointsDescriptors()
        scene_img: nd.array
            image of a scene where to looking for mathces
        threshold: float
            threshold for the ratio test
        min_matches: int
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
    if len(results.matches) < min_matches:
        return None

    # init accumulator
    acc= accumulator.Accumulator(scene_img,cell_size)

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

        scene_centroid =scene_kp.pt - scaled_vector
        acc.castVote(scene_centroid,scene_kp)

    # Find the maximum votes in the accumulator
    results.max_score,results.centroids= acc.getMax()


    return results

def find_instances(scene_paths, product_paths, threshold=constants.THRESHOLD, min_matches=constants.MIN_MATCHES):
    """
    find instances of the model images in the scene images using GHT and SIFT descriptors

    Parameters
    ----------
    scene_paths: image names in the constants.SCENES_PATH directory
    product_paths: image names in the constants.MODELS_PATH directory
    threshold=0.75 threshold for the ratio test
    min_matches=200: minimum number of mathces that need to be found in a scene

    Results
    ----------
    results: List[SceneAnalisys]
        list of of output for each scene
    """

    results = []

    # OFFLINE PHASE: compute keypoint,descriptors,vectors of the model images
    models = Model.get_models(product_paths)

    # ONLINE PHASE: run object detection on the scene images with the GHT + SIFT pipeline
    for scene_path in scene_paths:

        # read scene image
        scene_img = cv.imread(scene_path, cv.IMREAD_GRAYSCALE)
        scene_analisys = SceneAnalisys(scene_name=scene_path,scene=scene_img)

        for model in models:
            ghtOutput=  generalized_hough_transform(model, scene_img, threshold, min_matches)
            if ghtOutput!= None:
                modelFound = ModelFound(model,len(ghtOutput.centroids),centroids=ghtOutput.centroids,matches=ghtOutput.matches)
                modelFound.bboxes=Bbox.find_bboxes(model,ghtOutput.scene_keypoints,ghtOutput.centroids,ghtOutput.matches )
                scene_analisys.model_instances.append(modelFound)


        results.append(scene_analisys)

    return results
