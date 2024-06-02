import numpy as np
import cv2 as cv
from model import Model

class Bbox:

    @staticmethod
    def find_bboxes(model: Model, scene_keypoints: np.ndarray, matches: np.ndarray, min_match_threshold=15, max_distortion=1.4, bbox_overlap_threshold=0.5):
        '''
        find model boxes in the scene image using homography between model and scene keypoints

        Parameters
        ----------
        model: Model
            model image to search for boxes
        scene_keypoints: ndarray
            scene keypoints

        Returns
        ---------
        list of boxes that has been found on the scene image
        '''

        #bbox_props_list = []
        #bbox = Bbox(model)

        #if matches != None and len(matches)> min_match_threshold:

        #    src_pts= np.float32([model._keypoints[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        #    dst_pts= np.float32([scene_keypoints[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

        #    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC ) #RANSAC
        #    matchesMask = mask.ravel().tolist()

        #    bbox_transformed = bbox.transform_box(M)

        #    valid_shape = bbox_transformed.valid_bbox_shape(max_distortion)

        #    center = np.mean(bbox_transformed._corners, axis=0).flatten()

        #    if valid_shape:
        #        bbox_props_list.append({
        #            'model': model,
        #            'corners': bbox_transformed._corners,
        #            'center': center,
        #            'match_number': len(matches),
        #        })

        #    filtered_bboxes = filter_overlap(bbox_props_list, bbox_overlap_threshold)

        #return filtered_bboxes, matchesMask

    @staticmethod
    def filter_overlap(bbox_props_list: list['Bbox'], overlap_threshold: float):
        '''
        Filter boxes based on overlap thrashold

        '''
        filtered_bboxes = []
        for i, bbox1 in enumerate(bbox_props_list):
            keep = True
            for j, bbox2 in enumerate(bbox_props_list):
                if i != j:
                    overlap= bbox1.bbox_overlap(bbox2)
                    if overlap > overlap_threshold:
                        keep = False
            if keep:
                filtered_bboxes.append(bbox1)

        return filtered_bboxes

    @staticmethod
    def getModelKeypointsDescriptors(models_imgs:list[str]) -> list[Model]:
        """
        Loop trough model image names array and compute keypoints SIFT descriptors edge_vectors and centroid

        Parameters
        ----------
            model_imgs: array of image names in the constants.MODELS_PATH directory

        Returns
        -------
            models: list[Model]
               Models object extracted from the model image
        """
        models= []
        for model_img_name in models_imgs: models.append(Model(model_name=model_img_name))
        return models

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
        self.get_bbox_sides()
        self.get_bbox_diagonals()

    def get_bbox_sides(self):
        '''
        compute sides of the bounding box in this order: top, right, down, left.

        Parameters
        ----------
        bbox : array
            bounding box, constituted of an array of shape (n_corners, 1, 2).

        Returns
        -------
        l1, l2, l3, l4 : float
            edges of the bounding box.
        '''

        l1 = np.linalg.norm(self._corners[0] - self._corners[1])
        l2 = np.linalg.norm(self._corners[1] - self._corners[2])
        l3 = np.linalg.norm(self._corners[2] - self._corners[3])
        l4 = np.linalg.norm(self._corners[3] - self._corners[0])

        self.l1= l1
        self.l2= l2
        self.l3= l3
        self.l4= l4

    def get_bbox_diagonals(self):
        '''
        Compute diagonals of the bounding box in this order: max diagonal and min diagonal.

        '''

        d1 = np.linalg.norm(self._corners[0] - self._corners[2])
        d2 = np.linalg.norm(self._corners[1] - self._corners[3])
        self._d1 = max(d1, d2)
        self._d2 = min(d1, d2)

    def transform_box(self,M:np.ndarray  ):
        '''
        Transoform the bbox with a homography

        Parameters
        ----------
        M : np.ndarray
            matrix with which perform the transformation

        Return
        ------
        trasformed_bbox: Bbox
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

    def bbox_overlap(self, bbox: 'Bbox'):
        '''
        Compute the size of the intersection between 2 bounding boxes

        Parameters
        ----------
        bbox: Bbox
            Bounding box to compare

        Returns
        ----------
        result : float
            the dimension of the overlapping area
        '''
        # Convert bounding boxes to rectangles
        rect1 = cv.boundingRect(self._corners)
        rect2 = cv.boundingRect(bbox._corners)

        # Calculate intersection rectangle
        x1 = max(rect1[0], rect2[0])
        y1 = max(rect1[1], rect2[1])
        x2 = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
        y2 = min(rect1[1] + rect1[3], rect2[1] + rect2[3])

        if x1 < x2 and y1 < y2:
            intersection_area = (x2 - x1) * (y2 - y1)
            bbox1_area = rect1[2]*rect1[3]
            bbox2_area = rect2[2]* rect2[3]
            return intersection_area / float(bbox1_area + bbox2_area - intersection_area)
        else:
            return 0
