import cv2 as cv
import numpy as np
from lib.ght_sift import SceneAnalisys

def printCentroids(sceneAnalisys: SceneAnalisys):
    '''
    Print scene images with model and centroids

    Parameters
    scene_img: np.ndarray
    '''

    # print centroids in the image
    finalImg = np.copy(sceneAnalisys.scene)
    for modelFound in sceneAnalisys.model_instances:
        finalImg = np.copy(sceneAnalisys.scene)
        for centroid in modelFound.centroids:
            finalImg=cv.circle(finalImg,(int(centroid[0]),int(centroid[1])),thickness=10,radius=10,color=(0,0,255))
        cv.imshow('model image', modelFound.model._model_img)
        cv.imshow('detect instance', finalImg)
        cv.waitKey(0)
        cv.destroyAllWindows()

def printSceneAnalisys(sceneAnalisys: SceneAnalisys):
    '''
    Print scene images with model and centroids

    Parameters
    scene_img: np.ndarray
    '''

    # print centroids in the image
    finalImg = np.copy(sceneAnalisys.scene)
    for modelFound in sceneAnalisys.model_instances:
        finalImg = np.copy(sceneAnalisys.scene)
        for centroid in modelFound.centroids:
            finalImg=cv.circle(finalImg,(int(centroid[0]),int(centroid[1])),thickness=10,radius=10,color=(0,0,255))
        for bbox in modelFound.bboxes:
            cv.line(finalImg,(int(bbox._corners[0][0][0]),int(bbox._corners[0][0][1])), (int(bbox._corners[1][0][0]),int(bbox._corners[1][0][1])),(0,255,0), 4)
            cv.line(finalImg,(int(bbox._corners[1][0][0]),int(bbox._corners[1][0][1])), (int(bbox._corners[2][0][0]),int(bbox._corners[2][0][1])),(0,255,0), 4)
            cv.line(finalImg,(int(bbox._corners[2][0][0]),int(bbox._corners[2][0][1])), (int(bbox._corners[3][0][0]),int(bbox._corners[3][0][1])),(0,255,0), 4)
            cv.line(finalImg,(int(bbox._corners[3][0][0]),int(bbox._corners[3][0][1])), (int(bbox._corners[0][0][0]),int(bbox._corners[0][0][1])),(0,255,0), 4)
        cv.imshow('model image', modelFound.model._model_img)
        cv.imshow('detect instance', finalImg)
        cv.waitKey(0)
        cv.destroyAllWindows()

