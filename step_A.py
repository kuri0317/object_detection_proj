import cv2
import lib.constants as constants
from lib.model import *

# model images
scene_paths = [
    constants.SCENES_PATH + '/e1.png',
    constants.SCENES_PATH +'/e2.png',
    constants.SCENES_PATH +'/e3.png',
    constants.SCENES_PATH +'/e4.png',
    constants.SCENES_PATH +'/e5.png'
]

product_paths = [
    constants.MODELS_PATH + '/0.jpg',
    constants.MODELS_PATH +'/1.jpg',
    constants.MODELS_PATH +'/11.jpg',
    constants.MODELS_PATH +'/19.jpg',
    constants.MODELS_PATH +'/24.jpg',
    constants.MODELS_PATH +'/26.jpg',
    constants.MODELS_PATH +'/25.jpg'
]

models = Model.get_models(product_paths)

for scene_img_path in scene_paths:

    scene_img = cv2.imread(scene_img_path, cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create()
    keypoints_scene, descriptors_scene = sift.detectAndCompute(scene_img, None)

    for model in models:

        # find matching using kdtree nearest neighbour search
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(model._descriptors, descriptors_scene, k=2)

        # ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < constants.THRESHOLD * n.distance:
                good_matches.append(m)

        # print matches
        img_matches = cv2.drawMatches(model._model_img, model._keypoints, scene_img, keypoints_scene, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('Matches', img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if len(good_matches) >= constants.STEP_A_MIN_MATCHES:
            print(f"instance {model._model_name} found")

        else:
            print(f"instance {model._model_name} not found")
