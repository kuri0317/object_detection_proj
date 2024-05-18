import numpy as np
import cv2 as cv
import constants
from utils import getModelKeypointsDescriptors
from utils import  compute_ght_SIFT

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

scenes = ['m1.png', 'm2.png', 'm3.png', 'm4.png', 'm5.png']
model_images = ['0.jpg', '1.jpg', '11.jpg', '19.jpg', '24.jpg', '26.jpg', '25.jpg']


    

def find_instances(scenes,model_images):
    
    sift = cv.SIFT_create()

    # compute descriptors on model images
    models = getModelKeypointsDescriptors(model_images)

    for scene in scenes:

        # read scene image file and compute keypoints and SIFT descriptor
        scene_img = cv.imread(constants.SCENES_PATH +'/' + scene, cv.IMREAD_GRAYSCALE)
        keypoints_scene, descriptors_scene = sift.detectAndCompute(scene_img, None)
        
        for model in models:

            compute_ght_SIFT(model=model,target_keypoints=keypoints_scene,target_descriptors=descriptors_scene,target_image_size=scene_img.shape)



            ########################OLD
            ## filter correspondencies based on distance thresholding
            #good_matches = []
            #for m, n in matches:
            #    if m.distance < threshold * n.distance:
            #        good_matches.append(m)
            #
            ## check if there are enough correspondencies between model image and target
            #if len(good_matches) >= min_matches:

            #    #scene_centroid = np.mean([keypoint.pt for keypoint in keypoints_scene], axis=0)
            #    #scene_vectors = [keypoint.pt -centroid for keypoint in keypoints_scene] #calcolo vettori e baricentri
            #   
            #    #run ght on the scene image
            #    ght= cv.createGeneralizedHoughBallard()
            #    ght.setTemplate(model['model_img'])
            #    positions, votes = ght.detect(scene_img)
            #    if positions != None:

            #        # Disegna
            #        scene_centroids = product.copy()
            #        cv.circle(model['model_img'], tuple(map(int, centroid)), 10, (0, 255, 0), -1)
            #            
            #        cv.imshow('scena con baricentri e vettori', scene_centroids)
            #        cv.waitKey(0)
            #        cv.destroyAllWindows()

            #        if len(edge_vectors)> max_vect : #controllare che non si stampino troppi vettori
            #            edge_vectors= edge_vectors[:max_vect]




            #    else:
            #        print ("rilevazione non valida")

            #    #if model['model_name'] in scene_count:
            #    #    scene_count[model['model_name']]['n_instance'] += 1
            #    #else:
            #    #    scene_count[model['model_name']] = {
            #    #        'n_instance': 1,
            #    #        'centroid': model['centroid'],
            #    #        'edge_vectors': model['edge_vectors']
            #    #    }
            #'''
            ## Disegna
            #scene_centroids = product.copy()
            #cv.circle(scene_centroids, tuple(map(int, centroid)), 10, (0, 255, 0), -1)
            #    
            #cv.imshow('scena con baricentri e vettori', scene_centroids)
            #cv.waitKey(0)
            #cv.destroyAllWindows()

            #if len(edge_vectors)> max_vect : #controllare che non si stampino troppi vettori
            #    edge_vectors= edge_vectors[:max_vect]
            #
            #'''
                

find_instances(scenes,model_images)


#for scene, scene_count in count.items():
#    print(f"Scena: {scene}")
#
#    for prod_img, c in scene_count.items():
#        print(f"Prodotto: {prod_img}")
#        print(f"Istanze trovate: {c['n_instance']}")
#        print(f"Baricentro del prodotto: {c['centroid']}")
#        print(f"Vettori degli edge al baricentro ( {len(c['edge_vectors'])} max vettori)")
#        for vector in c['edge_vectors']:
#            print(vector)
#        print("")  # Stampa una riga vuota per separare le informazioni dei prodotti
#    print("")  # Stampa una riga vuota per separare le informazioni delle scene

