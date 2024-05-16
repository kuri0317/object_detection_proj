import numpy as np
import cv2
from collections import defaultdict

def compute_keypoints_descriptors(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def compute_centroid(keypoints):
    points = np.array([kp.pt for kp in keypoints])
    centroid = np.mean(points, axis=0)
    return centroid

def compute_linking_vectors(keypoints, centroid):
    vectors = [kp.pt - centroid for kp in keypoints]
    return vectors

def generalized_hough_transform(model_img, scene_img, threshold=0.75):
    # Step 1: Compute keypoints and descriptors for the model image
    keypoints_model, descriptors_model = compute_keypoints_descriptors(model_img)
    centroid_model = compute_centroid(keypoints_model)
    linking_vectors = compute_linking_vectors(keypoints_model, centroid_model)
    
    # Step 2: Compute keypoints and descriptors for the scene image
    keypoints_scene, descriptors_scene = compute_keypoints_descriptors(scene_img)

    # Step 3: Match descriptors between model and scene
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors_model, descriptors_scene, k=2)
    
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
        
        model_kp = keypoints_model[model_idx]
        scene_kp = keypoints_scene[scene_idx]
        
        model_vector = linking_vectors[model_idx]
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

    for scene_path in scene_paths:
        scene_img = cv2.imread(scene_path, cv2.IMREAD_GRAYSCALE)
        scene_count = {}

        for prod_path in product_paths:
            product_img = cv2.imread(prod_path, cv2.IMREAD_GRAYSCALE)
            possible_centroids, max_votes = generalized_hough_transform(product_img, scene_img, threshold)

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

                if prod_path in scene_count:
                    scene_count[prod_path]['n_instance'] += 1
                else:
                    scene_count[prod_path] = {
                        'n_instance': 1,
                        'centroid': centroid,
                        'edge_vectors': edge_vectors
                    }

            else:
                print(f"Not enough good matches for product {prod_path} in scene {scene_path}")

        counts[scene_path] = scene_count

    return counts

# Example usage
scene_paths = ['object_detection_project/scenes/m1.png', 'object_detection_project/scenes/m2.png', 'object_detection_project/scenes/m3.png', 'object_detection_project/scenes/m4.png', 'object_detection_project/scenes/m5.png']
product_paths = ['object_detection_project/models/0.jpg', 'object_detection_project/models/1.jpg', 'object_detection_project/models/11.jpg', 'object_detection_project/models/19.jpg', 'object_detection_project/models/24.jpg', 'object_detection_project/models/26.jpg', 'object_detection_project/models/25.jpg']

counts = find_instances(scene_paths, product_paths, threshold=0.75, min_matches=150, max_vectors=5)

# Print results
for scene_path, scene_count in counts.items():
    print(f"Scene: {scene_path}")
    for prod_path, data in scene_count.items():
        print(f"Product: {prod_path}")
        print(f"Instances found: {data['n_instance']}")
        print(f"Product centroid: {data['centroid']}")
        print(f"Edge vectors: {data['edge_vectors']}")
