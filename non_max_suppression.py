
import numpy as np
import cv2


def non_max_suppression (boxes, overlapThresh):#per rilevazioni duplicate vicine

    if len(boxes)==0:
        return[]

    if boxes.dtype.kind =="i":
        boxes = boxes.astype("float")
    
    pick=[]
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    area =(x2-x1 +1)*(y2-y1+1)
    idxs=np.argsort(y2)

    while len(idxs)>0:
        last =len(idxs)-1
        i= idxs[last]
        pick.append(i)

        xx1 =np.maximum(x1[1], x1[idxs[:last]])
        yy1 =np.maximum(y1[1], y1[idxs[:last]])
        xx2 =np.minimum(x2[1], x2[idxs[:last]])
        yy2 =np.minimum(y2[1], y2[idxs[:last]])

        w= np.maximum(0, xx2-xx1 +1)
        h = np.maximum(0, yy2-yy1 +1)

        overlap =(w*h)/area[idxs[:last]]

        idxs= np.delete(idxs, np.concatenate(([last], np.where(overlap> overlapThresh)[0])))
    return boxes[pick].astype("int")
