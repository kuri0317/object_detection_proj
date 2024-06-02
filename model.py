import numpy as np
import cv2 as cv
class Model:

    _keypoints: tuple
    _descriptors: np.ndarray
    _model_img: np.ndarray
    _centroid: np.ndarray
    _vectors: list
    _model_name: str

    def __init__(self,model_name: str):
        """
        create the model object SIFT descriptors edge_vectors and centroid of the models_img path

        Parameters
        ----------
            model_img: name of the image
        """
        self._model_name = model_name
        sift = cv.SIFT_create()

        # read model image
        self._model_img = cv.imread(model_name, cv.IMREAD_GRAYSCALE)

        # compute keypoints and SIFT descriptors
        self._keypoints, self._descriptors= sift.detectAndCompute(self._model_img, None)

        #compute centroid as the barycenter
        self._centroid = np.mean(np.array([kp.pt for kp in self._keypoints]), axis=0)

        # compute vectors from keypoints to barycenter
        self._vectors = [kp.pt - self._centroid for kp in self._keypoints]
