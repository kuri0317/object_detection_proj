import numpy as np
import cv2
from utils import getModelKeypointsDescriptors
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

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

#scene_2 = ['m1.png', 'm2.png', 'scenes/m3.png', 'm4.png', 'm5.png']
#product_2 = ['0.jpg', '1.jpg', '11.jpg', '19.jpg', '24.jpg', '26.jpg', '25.jpg']
scene_2 = ['object_detection_project/scenes/m1.png', 'object_detection_project/scenes/m2.png', 'object_detection_project/scenes/m3.png', 'object_detection_project/scenes/m4.png', 'object_detection_project/scenes/m5.png']
product_2 = ['object_detection_project/models/0.jpg', 'object_detection_project/models/1.jpg', 'object_detection_project/models/11.jpg', 'object_detection_project/models/19.jpg', 'object_detection_project/models/24.jpg', 'object_detection_project/models/26.jpg', 'object_detection_project/models/25.jpg']

def find_instances(scene_2, product_2, threshold=0.75, min_matches=150, max_vect=5):
    
    sift = cv2.SIFT_create()
    count = {}

    for scene in scene_2:

        scene_img = cv2.imread(scene, cv2.IMREAD_GRAYSCALE)
        keypoints_scene, descriptors_scene = sift.detectAndCompute(scene_img, None)
        scene_count ={}
    
        for prod_img in product_2:
            
            product = cv2.imread(prod_img, cv2.IMREAD_GRAYSCALE)
            keypoints_prod, descriptors_prod = sift.detectAndCompute(product, None)
            
            # Matcher metodo FLANN
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(descriptors_prod, descriptors_scene, k=2)

            good_matches = []
            for m, n in matches:
                if m.distance < threshold * n.distance:
                    good_matches.append(m)
        
            if len(good_matches) >= min_matches:

                centroid = np.mean([keypoint.pt for keypoint in keypoints_prod], axis=0)
                edge_vectors = [keypoint.pt -centroid for keypoint in keypoints_prod]
                scene_centroid = np.mean([keypoint.pt for keypoint in keypoints_scene], axis=0)
                scene_vectors = [keypoint.pt -centroid for keypoint in keypoints_scene] #calcolo vettori e baricentri
               
                #ght
                ght= cv2.createGeneralizedHoughBallard()
                ght.setTemplate(product)
                detections = ght.detect(scene_img)
              
                if detections is not None:

                    #serve accumulatore per voti? conviene fare funzione a parte?
                    votes = np.zeros_like( centroid )

                    for q_vec in edge_vectors:
                        for m_vec in scene_vectors:
                            # Calcola la trasformazione scalando il vettore modello in base al vettore di query
                            scale_factor = np.linalg.norm(q_vec) / np.linalg.norm(m_vec)
                            transformed_vector = m_vec * scale_factor
                            
                            # Calcola la posizione del baricentro traslato e vota
                            vote_position = scene_centroid + transformed_vector 
                            votes += vote_position
                            #vedi dove stampare i voti

                    scene_with_instances= cv2.cvtColor(scene_img, cv2.COLOR_GRAY2BGR)
                    for detection in detections:
                        for point in detection['pos']:
                            x, y = int(point[0]), int(point[1])
                            w, h = product.shape[::-1]
                            cv2.rectangle(scene_with_instances, (x, y), (x + w, y + h), (0, 255, 0), 2) #disegniamo i riquadri
                    
                    cv2.imshow('Scene with instances', scene_with_instances)
                    cv2.waitKey(0)

                else:
                    print ("rilevazione non valida")

                if prod_img in scene_count:
                    scene_count[prod_img]['n_instance'] += 1
                else:
                    scene_count[prod_img] = {
                        'n_instance': 1,
                        'centroid': centroid,
                        'edge_vectors': edge_vectors
                    }
           '''
            # Disegna
            scene_centroids = product.copy()
            cv2.circle(scene_centroids, tuple(map(int, centroid)), 10, (0, 255, 0), -1)
                
            cv2.imshow('scena con baricentri e vettori', scene_centroids)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            if len(edge_vectors)> max_vect : #controllare che non si stampino troppi vettori
                edge_vectors= edge_vectors[:max_vect]'''
                
        count [scene]= scene_count
    
    cv2.destroyAllWindows()
    return count    

count = find_instances(scene_2,product_2,threshold=0.75, min_matches=150, max_vect=5)


for scene, scene_count in count.items():
    print(f"Scena: {scene}")

    for prod_img, c in scene_count.items():
        print(f"Prodotto: {prod_img}")
        print(f"Istanze trovate: {c['n_instance']}")
        print(f"Baricentro del prodotto: {c['centroid']}")
        print(f"Vettori degli edge al baricentro ( {len(c['edge_vectors'])} max vettori)")
        for vector in c['edge_vectors']:
            print(vector)
        print("")  # Stampa una riga vuota per separare le informazioni dei prodotti
    print("")  # Stampa una riga vuota per separare le informazioni delle scene

