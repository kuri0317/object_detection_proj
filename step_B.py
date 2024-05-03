import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import getModelKeypointsDescriptors
import constants
'''
#STEP 2  
STEP 2 Multiple Instance Detection:
Test on scene image: {m1.png, m2.png, m3.png, m4.png, m5.png}
Use product images: {0.jpg, 1.jpg, 11.jpg, 19.jpg, 24.jpg, 26.jpg, 25.jpg}


In addition to what achieved at step A, the system should now be able to detect multiple instance of the
same product. Purposely, students may deploy local invariant feature together with the GHT (Generalized
Hough Transform). More precisely, rather than relying on the usual R-Table, the object model acquired at
training time should now consist in vectors joining all the features extracted in the model image to their
barycenter; then, at run time all the image features matched with respect to the model would cast votes
for the position of the barycenter by scaling appropriately the associated joining vectors (i.e. by the ratio of
sizes between the matching features).
'''

scene_2 = ['m1.png', 'm2.png', 'scenes/m3.png', 'm4.png', 'm5.png']
product_2 = ['0.jpg', '1.jpg', '11.jpg', '19.jpg', '24.jpg', '26.jpg', '25.jpg']
#scene_2 = ['object_detection_project/scenes/m1.png', 'object_detection_project/scenes/m2.png', 'object_detection_project/scenes/m3.png', 'object_detection_project/scenes/m4.png', 'object_detection_project/scenes/m5.png']
#product_2 = ['object_detection_project/models/0.jpg', 'object_detection_project/models/1.jpg', 'object_detection_project/models/11.jpg', 'object_detection_project/models/19.jpg', 'object_detection_project/models/24.jpg', 'object_detection_project/models/26.jpg', 'object_detection_project/models/25.jpg']

def find_instances(scene_2, product_2, threshold=0.75, min_matches=131):
    
    sift = cv2.SIFT_create()

    scene = cv2.imread(scene_2, cv2.IMREAD_GRAYSCALE)
    keypoints_scene, descriptors_scene = sift.detectAndCompute(scene, None)

    count = {}


    for prod_img in product_2:
        product = cv2.imread(prod_img, cv2.IMREAD_GRAYSCALE)

        keypoints_prod, descriptors_prod = sift.detectAndCompute(product, None)
        
        # Matcher descrittori metodo FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=300)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(descriptors_prod, descriptors_scene, k=2)
            #creare vettori
            #vectors_prod = np.array([keypoint.pt for keypoint in keypoints_prod])
            #vectors_scene = np.array([keypoint.pt for keypoint in keypoints_scene])
        
        good_matches = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good_matches.append(m)
        
        if len(good_matches) >= min_matches:
            count[product_2] = len(good_matches)
    
    return count    

for scenes in scene_2:
    print(f"scena: {scenes}")

    count = find_instances(scenes, product_2)

    for product_2,  instances in count.items():
        print(f"Prodotto: {product_2}, Istanze trovate: {instances}")
    
    print("") 
        #-----------------------------

'''
        def match_and_vote(scene_image, model_image_centroid, model_image_vectors):
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(scene_image, None)
            
            votes = np.zeros_like(model_image_centroid)
            
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(descriptors, descriptors_model, k=2)

            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    scene_point = keypoints[m.trainIdx].pt
                    model_point = model_image_vectors[m.queryIdx]
                    
                    size_ratio = m.distance / n.distance
                
                    scaled_vector = (model_point - model_image_centroid) * size_ratio
                    
                    votes += scaled_vector
            
            return votes
    
        
        model_centroid = np.mean(vectors_model, axis=0)
        scene_centroids = [] # lista per le posizioni

        bf = cv2.BFMatcher()

        matches = bf.knnMatch(descriptors_model, descriptors_scene, k=2)

        good_matches = [] 

        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                
        for m in good_matches:
            scene_centroid_estimate = vectors_scene[m.trainIdx] + model_centroid - vectors_model[m.queryIdx]
            scene_centroids.append(scene_centroid_estimate)

        #clustering oer raggruppare posizioni
        X = np.array(scene_centroids)
        db = DBSCAN(eps=10, min_samples=5).fit(X)
        labels = db.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print("Numero istanze multiple trovate:", n_clusters)

        # Disegna
        scene_img_with_centroids = scene.copy()
        for i in range(n_clusters):
            cluster_points = X[labels == i]
            centroid = np.mean(cluster_points, axis=0)
            cv2.circle(scene_img_with_centroids, tuple(map(int, centroid)), 10, (0, 255, 0), -1)
            for point in cluster_points:
                cv2.arrowedLine(scene_img_with_centroids, tuple(map(int, point)), tuple(map(int, centroid)), (0, 0, 255), 2)

        cv2.imshow('scena con baricentri e vettori', scene_img_with_centroids)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

'''
#----------------------------------------
