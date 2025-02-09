import numpy as np
import cv2 as cv
from lib.model import Model

# class representation of a cereal bounding box with utility methods
class Bbox:

    # ATTRIBUTES
    _corners: np.ndarray
    _l1: np.ndarray
    _l2: np.ndarray
    _l3: np.ndarray
    _l4: np.ndarray
    _d1: np.ndarray
    _d2: np.ndarray
    _model: Model

    def __init__(self, model: Model):
        """
        create the bbox object by the model object

        Parameters
        ----------
        model: Model
            model object
        """
        self._model = model
        h, w = model._model_img.shape
        self._corners = np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2).astype(np.float32)

        # sides
        self._l1= np.linalg.norm(self._corners[0] - self._corners[1])
        self._l2= np.linalg.norm(self._corners[1] - self._corners[2])
        self._l3= np.linalg.norm(self._corners[2] - self._corners[3])
        self._l4= np.linalg.norm(self._corners[3] - self._corners[0])

        # diagonals
        d1 = np.linalg.norm(self._corners[0] - self._corners[2])
        d2 = np.linalg.norm(self._corners[1] - self._corners[3])
        self._d1 = max(d1, d2)
        self._d2 = min(d1, d2)

    @staticmethod
    def find_bboxes(model: Model, scene_kp:np.ndarray, centroids: np.ndarray, matches: np.ndarray, max_distortion=1.4, bbox_overlap_threshold=0.5):
        '''
        find model boxes in the scene image using homography between model and scene keypoints

        Parameters
        ----------
        model: Model
            model image to search for boxes
        centroids: ndarray
            scene keypoints

        Returns
        ---------
        list of boxes that has been found on the scene image
        '''

        bbox_props_list = []
        bbox = Bbox(model)

        for centroid in centroids:
            dst_points=[]
            src_points=[]
            for m in matches:
                if scene_kp[m.trainIdx] in centroid[2]:
                    src_points.append(model._keypoints[m.queryIdx].pt)
                    dst_points.append(scene_kp[m.trainIdx].pt)
            src_points= np.float32(src_points).reshape(-1,1,2)
            dst_points= np.float32(dst_points).reshape(-1,1,2)

            if len(src_points) >= 4 and len(dst_points) >= 4:

                M,_= cv.findHomography(src_points, dst_points, cv.RANSAC )
                if M is not None:
                    bbox_transformed = bbox.transform_box(M)
                    if bbox_transformed.valid_bbox_shape():
                        bbox_props_list.append(bbox_transformed)

        return bbox_props_list

    def transform_box(self,M:np.ndarray  ):
        '''
        Transoform the bbox with a homography

        Parameters
        ----------
        M : np.ndarray
            matrix with which perform the transformation

        Return
        ------
        transformed_bbox: Bbox
            transformed bounding box
        '''

        corners = cv.transform(self._corners, M)
        transformed_bbox = Bbox(self._model)
        transformed_bbox._corners = corners

        return transformed_bbox

    def valid_bbox_shape(self, max_distortion=1.4):
        '''
        Check if the bbox is valid by controlling the ratio between diagonal and sides of the bbox with the max_distortion

        Parameters
        ----------
        bbox : array
            bounding box, constituted of an array of shape (n_corners, 1, 2).
        max_distortion: float
            max accepted distortion, default 1.4

        Returns
        ----------
        result : bool
            if the bbox is or not distorted
        '''

        valid_diagonal = self._d1 / self._d2 <= max_distortion
        valid_edges1 = max(self._l1, self._l3) / min(self._l1, self._l3) <= max_distortion
        valid_edges2 = max(self._l2, self._l4) / min(self._l2, self._l4) <= max_distortion

        return valid_diagonal and valid_edges1 and valid_edges2
