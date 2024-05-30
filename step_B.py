import numpy as np
import cv2 as cv
import constants
from collections import defaultdict
from utils import  getModelKeypointsDescriptors
from ght_sift import  generalized_hough_transform
import argparse

# parameter parser function
def getParams():
    parser = argparse.ArgumentParser(prog='object_detection_project',description='box detection project',epilog='credits carnivuth,kuri')
    parser.add_argument('-t','--threshold',default=constants.THRESHOLD,help='thrashold for ratio test',type=float)
    parser.add_argument('-m','--minMatches',default=constants.MIN_MATCHES,help='minimum number of matches for detecting the model in the target image',type=int)
    return parser.parse_args()

''''''
def get_bbox_edges(bbox):
    l1 = np.linalg.norm(bbox[0] - bbox[1])
    l2 = np.linalg.norm(bbox[1] - bbox[2])
    l3 = np.linalg.norm(bbox[2] - bbox[3])
    l4 = np.linalg.norm(bbox[3] - bbox[0])
    return l1, l2, l3, l4


def get_bbox_diagonals(bbox):
    d1 = np.linalg.norm(bbox[0] - bbox[2])
    d2 = np.linalg.norm(bbox[1] - bbox[3])
    return max(d1, d2), min(d1, d2)

def valid_bbox_shape(bbox, max_distortion=1.4):

    l1, l2, l3, l4 = get_bbox_edges(bbox)
    d1, d2 = get_bbox_diagonals(bbox)

    valid_diagonal = d1 / d2 <= max_distortion
    valid_edges1 = max(l1, l3) / min(l1, l3) <= max_distortion
    valid_edges2 = max(l2, l4) / min(l2, l4) <= max_distortion

    return valid_diagonal and valid_edges1 and valid_edges2

def filter_overlap(bbox_props_list, overlap_threshold):
    def bbox_overlap(bbox1, bbox2):
       
        # Convert bounding boxes to rectangles
        rect1 = cv.boundingRect(np.int32(bbox1))
        rect2 = cv.boundingRect(np.int32(bbox2))

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
    
    filtered_bboxes = []
    for i, bbox1 in enumerate(bbox_props_list):
        if bbox1['valid_bbox']:
            keep = True
            for j, bbox2 in enumerate(bbox_props_list):
                if i != j and bbox2['valid_bbox']:
                    overlap= bbox_overlap(bbox1['corners'], bbox2['corners'])
                    if overlap > overlap_threshold:
                        if bbox1['match_number'] < bbox2['match_number']:
                            keep = False
                            break
            if keep:
                filtered_bboxes.append(bbox1)
    return filtered_bboxes
''''''
def find_bboxes(scene_img, model, scene_keypoints, matches, min_match_threshold=15, max_distortion=1.4, bbox_overlap_threshold=0.5):
    
    bbox_props_list = []
    h, w = model['model_img'].shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
#IMPLEMENTA RANSAC
    if matches != None and len(matches)> min_match_threshold:
        
        src_pts= np.float32([model['keypoints'][m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts= np.float32([scene_keypoints[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC , 5.0) #RANSAC
        matchesMask = mask.ravel().tolist()
        dst= cv.perspectiveTransform(pts, M)

        high_kp = True  #
        valid_shape = valid_bbox_shape(dst, max_distortion)  # 

        center = np.mean(dst, axis=0).flatten()
        
        bbox_props_list.append({
            'model': 'model_name',  #
            'corners': dst,
            'center': center,
            'match_number': len(matches),  #
            'sufficient_matches': high_kp,
            'valid_shape': valid_shape,
            'valid_bbox': high_kp and valid_shape,
        })

        filtered_bboxes = filter_overlap(bbox_props_list, bbox_overlap_threshold)

    return filtered_bboxes, matchesMask

def display_results(scene_img, model, matches, scene_keypoints, bbox_props_list, matchesMask):

    instance_count= len(bbox_props_list)
    scene_with_instances= cv.cvtColor(scene_img, cv.COLOR_GRAY2BGR)

    for bbox in bbox_props_list:
        corners= np.int32(bbox['corners'])
        if bbox['valid_bbox']:
            cv.polylines(scene_with_instances, [corners], True, (0,255,0),2)

    final = cv.drawMatches(model['model_img'], model['keypoints'], scene_with_instances, scene_keypoints, matches, None, matchesMask= matchesMask)

    print (f'number of instances found with RANSAC: {instance_count}')
    cv.imshow('detect instance', final)
    cv.waitKey(0)
    cv.destroyAllWindows()
        
def find_instances(scene_paths, product_paths, threshold=constants.THRESHOLD, min_matches=constants.MIN_MATCHES):
    """
    find instances of the model images in the scene images using GHT and SIFT descriptors

    Parameters:
    scene_paths: image names in the constants.SCENES_PATH directory
    product_paths: image names in the constants.MODELS_PATH directory 
    threshold=0.75 threshold for the ratio test
    min_matches=200: minimum number of mathces that need to be found in a scene
    """

    counts = {}

    # OFFLINE PHASE: compute keypoint,descriptors,vectors of the model images
    models = getModelKeypointsDescriptors(product_paths)
    
    # ONLINE PHASE: run object detection on the scene images with the GHT + SIFT pipeline
    for scene_path in scene_paths:

        scene_count = {}
        print(f"SEARCHING in Scene: {scene_path}")

        # read scene image
        scene_img = cv.imread(constants.SCENES_PATH + '/' + scene_path, cv.IMREAD_GRAYSCALE)

        for model in models:
            possible_centroids, max_votes, scene_keypoints, scene_descriptors,matches= generalized_hough_transform(model, scene_img, threshold, min_matches)

            # compute scale between model image and scene image
            scale_w = scene_img.shape[0] /model['model_img'].shape[0]
            scale_h = scene_img.shape[1]/ model['model_img'].shape[1]

            if matches is not None :
                bbox_props_list, matchesMask = find_bboxes(scene_img, model, scene_keypoints, matches)

                for bbox_props in bbox_props_list:
                    if model['model_name'] in scene_count:
                        scene_count[model['model_name']]['n_instance'] += 1
                    else:
                        scene_count[model['model_name']] = {'n_instance': 1}

                display_results(scene_img, model, matches , scene_keypoints, bbox_props_list, matchesMask)
                print(f'numero istanze trovate con box e ransac:{scene_count}')
                '''
#--------------------------------------
            if possible_centroids is not None:
                filtered_bboxes = find_bboxes(scene_img, model, scene_keypoints, matches, min_match_threshold=15, max_distortion=1.4, bbox_overlap_threshold=0.5)
                
                if  matches != None and len(matches)> min_matches :

                    src_pts= np.float32([model['keypoints'][m.queryIdx].pt for m in matches]).reshape(-1,1,2)
                    dst_pts= np.float32([scene_keypoints[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    #IMPLEMENTAZIONE RANSAC
                    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC , 5.0) #RANSAC
                    matchesMask = mask.ravel().tolist()
                    h, w = model['model_img'].shape 
                    pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)
                    dst = cv.perspectiveTransform(pts, M)

                    print (f'number of instances found with RANSAC: {scene_count}')
                
                if filtered_bboxes:
                    scene_with_instances = cv.cvtColor(scene_img, cv.COLOR_GRAY2BGR)
                    for bbox in filtered_bboxes:
                        corners = np.int32(bbox['corners'])
                        if bbox['valid_bbox']:
                            cv.polylines(scene_with_instances, [corners], True, (0, 255, 0), 2)
                    
                    final_image = cv.drawMatches(model['model_img'], model['keypoints'], scene_with_instances, scene_keypoints, matches, None, matchesMask=matchesMask)
      
                    cv.imshow("Detected Instances dentro filtered BOXXES", final_image)
                    cv.waitKey(0)
                    cv.destroyAllWindows()

                print(f"numero istanze trovate metodo box {scene_count}")

                scene_with_instances= cv.cvtColor(scene_img, cv.COLOR_GRAY2BGR)
#-------------------------------
                # draw bounding box for each possible centroid
                for centroid_pos in possible_centroids:

                    center_x, center_y = centroid_pos

                    w, h =  model['model_img'].shape[::-1]
                    w = w * scale_w
                    h = h * scale_h

                    starting_point= (int(center_x-(w/2)),int(center_y-(h/2)))

                    ending_point= (int(center_x+(w/2)),int(center_y+(h/2)))
                    
                    cv.rectangle(scene_with_instances,starting_point,ending_point,(0, 255, 0), 2)

                    final_image=cv.drawMatches(model['model_img'],model['keypoints'],scene_with_instances, scene_keypoints, matches,None)
                    
                cv.imshow('Scene with instances', final_image)
                cv.waitKey(0)
                cv.destroyAllWindows()

                print(f"numero istanze trovate metodo vecchio {scene_count}")'''
        #print(f"numero istanze trovate {scene_count}")
        counts[scene_path] = scene_count

    return counts

# main
scene_paths = ['m1.png', 'm2.png', 'm3.png', 'm4.png', 'm5.png']
product_paths = ['0.jpg', '1.jpg', '11.jpg', '19.jpg', '24.jpg', '26.jpg', '25.jpg']

args = getParams()
counts = find_instances(scene_paths, product_paths, args.threshold, args.minMatches)

