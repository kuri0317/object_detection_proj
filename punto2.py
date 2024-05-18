import numpy as np
import cv2 as cv
import constants
from collections import defaultdict
from utils import  getModelKeypointsDescriptors

def generalized_hough_transform(model, scene_img, threshold=0.75):
    """
    Compute ght alghorithm using SIFT descriptors

    Parameters:
    model: dict object as the output of getModelKeypointsDescriptors()
    scene_img: path to a scene in the constants.SCENE_PATH directory
    """

    # Step 2: Compute keypoints and descriptors for the scene image
    sift = cv.SIFT_create()
    keypoints_scene, descriptors_scene = sift.detectAndCompute(scene_img,None)

    # Step 3: Match descriptors between model and scene
    bf = cv.BFMatcher(cv.NORM_L2)
    matches = bf.knnMatch(model['descriptors'], descriptors_scene, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)
    
    print(f"Number of good matches: {len(good_matches)}")

    if len(good_matches) == 0:
        return None, 0

    # Step 4: Accumulator for votes

    accumulator = defaultdict(int)
    
    for match in good_matches:
        model_idx = match.queryIdx
        scene_idx = match.trainIdx
        
        model_kp = model['keypoints'][model_idx]
        scene_kp = keypoints_scene[scene_idx]
        
        model_vector = model['vectors'][model_idx]
        scale = scene_kp.size / model_kp.size
        rotated_vector = scale * model_vector
        
        scene_centroid = np.array(scene_kp.pt) - rotated_vector
        accumulator[tuple(scene_centroid)] += 1

    # Step 5: Find the position with the highest votes
    if accumulator:
        max_votes = max(accumulator.values())
        possible_centroids = [pos for pos, votes in accumulator.items() if votes == max_votes]

        return possible_centroids, max_votes
    else:
        return None, 0

def find_instances(scene_paths, product_paths, threshold=0.75, min_matches=150, max_vectors=5):
    counts = {}
    models = getModelKeypointsDescriptors(product_paths)


    
    for scene_path in scene_paths:
        scene_img = cv.imread(constants.SCENES_PATH + '/' + scene_path, cv.IMREAD_GRAYSCALE)
        scene_count = {}

        for model in models:
            possible_centroids, max_votes = generalized_hough_transform(model, scene_img, threshold)

            if max_votes >= min_matches:

                if len(model['vectors']) > max_vectors:
                    model['vectors'] = model['vectors'][:max_vectors]

                scene_with_instances = cv.cvtColor(scene_img, cv.COLOR_GRAY2BGR)
                #print(possible_centroids)
                for point in possible_centroids:
                    x, y = int(point[0]), int(point[1])
                    w, h = model['model_img'].shape[::-1]
                    cv.rectangle(scene_with_instances, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv.imshow('Scene with instances', scene_with_instances)
                cv.waitKey(0)
                cv.destroyAllWindows()

                if prod_path in scene_count:
                    scene_count[prod_path]['n_instance'] += 1
                else:
                    scene_count[prod_path] = {
                        'n_instance': 1,
                        'centroid': model['centroid'],
                        'vectors': model['vectors']
                    }

            else:
                print(f"Not enough good matches for product {model['model_name']} in scene {scene_path}")

        counts[scene_path] = scene_count

    return counts

# Example usage
scene_paths = ['m1.png', 'm2.png', 'm3.png', 'm4.png', 'm5.png']
product_paths = ['0.jpg', '1.jpg', '11.jpg', '19.jpg', '24.jpg', '26.jpg', '25.jpg']

counts = find_instances(scene_paths, product_paths, threshold=0.75, min_matches=150, max_vectors=5)

# Print results
for scene_path, scene_count in counts.items():
    print(f"Scene: {scene_path}")
    for prod_path, data in scene_count.items():
        print(f"Product: {prod_path}")
        print(f"Instances found: {data['n_instance']}")
        print(f"Product centroid: {data['centroid']}")
        print(f"Edge vectors: {data['edge_vectors']}")
