import cv2 as cv
import constants
def getModelKeypointsDescriptors(models_imgs):
    models= []
    for model_img_name in models_imgs:
        model_img = cv.imread(constants.MODELS_PATH +'/'+ model_img_name, cv.IMREAD_GRAYSCALE)
        sift = cv.SIFT_create()
        keypoints, descriptors= sift.detectAndCompute(model_img, None)

        models.append({
                "keypoints":keypoints,
                "descriptors":descriptors,
                "model_img":model_img,
                "model_name":model_img_name
                })
        
    return models
