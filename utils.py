import cv2 as cv
import constants
import numpy as np
def getModelKeypointsDescriptors(models_imgs):
    """
    Loop trough model image names array and compute keypoints SIFT descriptors edge_vectors and centroid

    Parameters:
        model_imgs: array of image names in the constants.MODELS_PATH directory
    Returns:
        models: array of dicts described as follows
                "keypoints":keypoints detected,
                "descriptors":descriptors computed,
                "model_img":model_img opencv image,
                "centroid":centroid computed,
                "edge_vectors":edge_vectors computed,
                "model_name":model_img_name name of the image in the constants.MODELS_PATH directory
    """
    models= []
    sift = cv.SIFT_create()

    for model_img_name in models_imgs:

        # read model image
        model_img = cv.imread(constants.MODELS_PATH +'/'+ model_img_name, cv.IMREAD_GRAYSCALE)

        # compute keypoints and SIFT descriptors
        keypoints, descriptors= sift.detectAndCompute(model_img, None)

        #compute centroid as the barycenter
        centroid = np.array((np.mean([keypoint.pt[0] for keypoint in keypoints]),np.mean([keypoint.pt[1] for keypoint in keypoints])))
        #print(centroid)

        # compute vectors from keypoints to barycenter
        edge_vectors = [centroid -keypoint.pt for keypoint in keypoints]

        models.append({
                "keypoints":keypoints,
                "descriptors":descriptors,
                "model_img":model_img,
                "centroid":centroid,
                "edge_vectors":edge_vectors,
                "model_name":model_img_name
                })
        
    return models

def compute_ght_SIFT(model,target_keypoints,target_descriptors,target_image_size):
    """
    compute GHT alghorithm with local invariant features using SIFT
    
    Parameters:
        model: one of the elements returned by the getModelKeypointsDescriptors function
        target_keypoints: keypoint of the target image
        target_descriptors: descriptors of the target image
        target_image_size: shape of the target image for the AA matrix
    """

    # find correspondencies with the FLANN NN search algorithm
    index_params = dict(algorithm=constants.FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(model['descriptors'], target_descriptors, k=1)

    ## init accumulator array AA
    AA = np.zeros(target_image_size)
    print(f'shape of accumulator array: {AA.shape}')

    for match in matches:
        # filter matches 
        # TODO

        # compute centroid vote
        #print(type(match[0]))
        centroid_vote= (np.round(target_keypoints[match[0].trainIdx].pt + model['edge_vectors'][match[0].queryIdx])).astype(int)
        print(f'vote casted by the match: {centroid_vote}')

        # cast vote in the accumulator array
        #AA[centroid_vote] += 1

    #print(AA)
