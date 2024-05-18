import cv2 as cv
import constants
import numpy as np
from collections import defaultdict
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
    print(f"Product: {model['model_name']}")
    # find correspondencies with the FLANN NN search algorithm
    index_params = dict(algorithm=constants.FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(model['descriptors'], target_descriptors, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"Number of good matches: {len(good_matches)}")

    if len(good_matches) == 0:
        return None, 0
    

    ## init accumulator array AA
    #AA = np.zeros(target_image_size)
    #print(f'shape of accumulator array: {AA.shape}')

    accumulator = defaultdict(int)
    for match in good_matches:
        # filter matches 
        # TODO
        model_idx = match.queryIdx
        scene_idx = match.trainIdx
        
        model_kp = (model['keypoints'][model_idx])
        scene_kp = target_keypoints[scene_idx]
        
        linking_vectors = [kp.pt - model['centroid'] for kp in model['keypoints']]
        model_vector = linking_vectors[model_idx]
        scale = scene_kp.size / model_kp.size
        rotated_vector = scale * model_vector
        
        scene_centroid = np.array(scene_kp.pt) - rotated_vector
        accumulator[tuple(scene_centroid)] += 1

        if accumulator:
            max_votes = max(accumulator.values())
            possible_centroids = [pos for pos, votes in accumulator.items() if votes == max_votes]

            return possible_centroids, max_votes
        else:
            return None, 0
        '''
    min_matches=200
    if max_votes >= min_matches:
                centroid = compute_centroid(compute_keypoints_descriptors(product_img)[0])
                edge_vectors = compute_linking_vectors(compute_keypoints_descriptors(product_img)[0], centroid)

                if len(edge_vectors) > max_vectors:
                    edge_vectors = edge_vectors[:max_vectors]

                scene_with_instances = cv2.cvtColor(scene_img, cv2.COLOR_GRAY2BGR)
                for point in possible_centroids:
                    x, y = int(point[0]), int(point[1])
                    w, h = product_img.shape[::-1]
                    cv2.rectangle(scene_with_instances, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.imshow('Scene with instances', scene_with_instances)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                '''
        # compute centroid vote
        #print(type(match[0]))
        #centroid_vote= (np.round(target_keypoints[match[0].trainIdx].pt + model['edge_vectors'][match[0].queryIdx])).astype(int)
        #print(f'vote casted by the match: {centroid_vote}')

        # cast vote in the accumulator array
        #AA[centroid_vote] += 1

    #print(AA)
    
