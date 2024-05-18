import cv2 as cv
import constants
import numpy as np

def getModelKeypointsDescriptors(models_imgs):
    """
    Loop trough model image names array and compute keypoints SIFT descriptors vectors and centroid

    Parameters:
        model_imgs: array of image names in the constants.MODELS_PATH directory
    Returns:
        models: array of dicts described as follows
                "keypoints":keypoints detected,
                "descriptors":descriptors computed,
                "model_img":model_img opencv image,
                "centroid":centroid computed,
                "vectors":vectors computed,
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
        centroid = np.mean(np.array([kp.pt for kp in keypoints]), axis=0)


        # compute vectors from keypoints to barycenter
        vectors = [kp.pt - centroid for kp in keypoints]

        models.append({
                "keypoints":keypoints,
                "descriptors":descriptors,
                "model_img":model_img,
                "centroid":centroid,
                "vectors":vectors,
                "model_name":model_img_name
                })
        
    return models
