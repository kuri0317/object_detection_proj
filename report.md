# OBJECT DETECTION PROJECT REPORT

as shown in the requirements the goal of the project is to detect istances of a product given as model in a multiple sets of target images.

## STEP 1

### PROJECT ANALISYS

For the first set of images, given the fact that only one instance of the model was present in the target image, the team decided to adopt an approach based on local invariant features and a kdtree based matching process

### IMPLEMENTATION

The system is implemented with a 2 step approach, in the first phase keypoints and sift descriptor are computed from the model images and in the second step model descriptor are used to find correspondencies in the target image using BFF search algorithm. keypoints detection and descriptor computation and matching are performed trough the use of the opencv

## STEP 2

### PROJECT ANALISYS

For the second set of images, in order to detect multiple instances of the same model,[generalized hough transform](https://carnivuth.github.io/computer_vision/pages/object_detection/GENERALIZED_HUGH_TRANSFORM) with [SIFT](https://carnivuth.github.io/computer_vision/pages/local_features/SIFT_DESCRIPTOR) descriptors was deployed

### IMPLEMENTATION

The implementation of GHT present in the [opencv library](https://docs.opencv.org/3.4/dc/d46/classcv_1_1GeneralizedHoughBallard.html) is not capable of doing comparison using SIFT descriptors, in order to address this issue the team decided to implement the GHT algorithm.

The implementation relies on the opencv functions in order to compute keypoints and SIFT descriptors.Then a voting process is performed using a bidimensional accumulator array, for each match the baricenter is computed using the scaled distance vector computed on the model

The accumulator array cast votes based on the barycenter coordinates, in order to increase resistance against noise, the accumulator has a tunable parameter to resize the single cell dimension

For each cell of the accumulator a list of keypoints that casted it is stored, in order to compute the homography later for the bounding box

The accumulator that returns the most voted barycenter, to improve robustness a tunable parameter to allow lower voted values is implemented

for each barycenter, the bounding box is computed exploiting opencv omographies funcitons, and the bounding box is than transposed on the computed target

## CONCLUSIONS

The GHT implementation results not so robust to noise, and a fine tuning is required in order to detect multiple bounding boxes in the image.
