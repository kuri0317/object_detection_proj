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

def generalized_hough_transform(keypoints_scene, keypoints_prod, descriptors_scene, descriptors_prod, centroid, threshold=0.75):
    # FLANN matcher
    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(descriptors_prod, descriptors_scene, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)
    print(f"Number of good matches: {len(good_matches)}")

    # Build accumulator
    accumulator = defaultdict(int)
    for match in good_matches:
        kp_scene = keypoints_scene[match.trainIdx]
        kp_prod = keypoints_prod[match.queryIdx]

        # Calculate the position of the centroid in the scene image
        dx = kp_scene.pt[0] - kp_prod.pt[0]
        dy = kp_scene.pt[1] - kp_prod.pt[1]

        # Accumulate votes
        pos = (int(dx + centroid[0]), int(dy + centroid[1]))
        accumulator[pos] += 1
     

    # Find the maximum votes in the accumulator
    if accumulator:
        max_votes = max(accumulator.values())
        possible_centroids = [pos for pos, votes in accumulator.items() if votes == max_votes]

        return possible_centroids, max_votes
    else:
        return None, 0

def find_instances(scene_paths, product_paths, threshold=0.75, min_matches=200, max_vectors=10):
    counts = {}

    for scene_path in scene_paths:
        print(f"searching in Scene: {scene_path}")
        scene_img = cv2.imread(scene_path, cv2.IMREAD_GRAYSCALE)
        keypoints_scene, descriptors_scene= compute_keypoints_descriptors(scene_img)

        scene_count = {}

        for prod_path in product_paths:
            print(f"analizzando Prodotto: {prod_path}")
            product_img = cv2.imread(prod_path, cv2.IMREAD_GRAYSCALE)
            keypoints_prod, descriptors_prod= compute_keypoints_descriptors(product_img)
            cv2.imshow('prodotto cercato', product_img)
            cv2.waitKey(0)
            if len(keypoints_prod)>=min_matches:

                centroid=compute_centroid(keypoints_prod)
                edge_vectors=compute_linking_vectors(keypoints_prod, centroid)
#GHT
                possible_centroids, max_votes = generalized_hough_transform(keypoints_scene, keypoints_prod, descriptors_scene, descriptors_prod, centroid, threshold)

                if possible_centroids:
                    scene_with_instances= cv2.cvtColor(scene_img,cv2.COLOR_GRAY2BGR)

                    for centroid_pos in possible_centroids:
                        x, y = centroid_pos
                        w, h = product_img.shape[::-1]
                        cv2.rectangle(scene_with_instances, (x-w //2 ,y-h// 2),(x+w// 2, y+h //2 ) , (0, 255, 0), 2)
                    
                    cv2.imshow('Scene with instances', scene_with_instances)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else: 
                    print(f"detection fail")

                    #update
                    if prod_path in scene_count:
                        scene_count[prod_path]['n_instance'] += 1
                    else:
                        if len(edge_vectors)>max_vectors:
                            edge_vectors=edge_vectors[:max_vectors]
                        scene_count[prod_path] = {
                            'n_instance': 1,
                            'centroid': centroid,
                            'edge_vectors': edge_vectors
                        }
                    print(f"numero istanze trovate {scene_count}")
        counts[scene_path] = scene_count

    return counts

# Example usage
scene_paths = ['object_detection_project/scenes/m1.png', 'object_detection_project/scenes/m2.png', 'object_detection_project/scenes/m3.png', 'object_detection_project/scenes/m4.png', 'object_detection_project/scenes/m5.png']
product_paths = ['object_detection_project/models/0.jpg', 'object_detection_project/models/1.jpg', 'object_detection_project/models/11.jpg', 'object_detection_project/models/19.jpg', 'object_detection_project/models/24.jpg', 'object_detection_project/models/26.jpg', 'object_detection_project/models/25.jpg']

counts = find_instances(scene_paths, product_paths, threshold=0.75, min_matches=200, max_vectors=10)

# Print results
for scene_path, scene_count in counts.items():
    print(f"Scene: {scene_path}")
    for prod_path, data in scene_count.items():
        print(f"Product: {prod_path}")
        print(f"Instances found: {data['n_instance']}")
        print(f"Product centroid: {data['centroid']}")
        print(f"Edge vectors: {data['edge_vectors']}")
