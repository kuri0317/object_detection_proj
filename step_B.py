#!/usr/bin/python
import argparse
import lib.constants as constants
from lib.ght_sift import  find_instances
from lib.print_utils import printSceneAnalisys

# GET ARGUMENTS
def getParams():
    parser = argparse.ArgumentParser(prog='DetectMeals',description='meals detection project',epilog='credits carnivuth kuri')
    parser.add_argument(
        '-m',
        '--minMatches',
        default=constants.MIN_MATCHES,
        help='minimum number of matches to detect in the scene image',
        type=int
    )
    parser.add_argument(
        '-f',
        '--flannIndexKdtree',
        default=constants.FLANN_INDEX_KDTREE,
        help='',
        type=int
    )
    parser.add_argument(
        '-t',
        '--threshold',
        default=constants.THRESHOLD,
        help='threshold for distance between matches',
        type=float
    )
    parser.add_argument(
        '-c',
        '--cellSize',
        default=constants.CELL_SIZE,
        help='ght accumulator for cell size',
        type=int
    )
    parser.add_argument(
        '-v',
        '--valuesFromMax',
        default=constants.VALUES_FROM_MAX,
        help='number of values lower than max to consider when computing the most voted centroid in the ght process',
        type=int
    )
    return parser.parse_args()

# main
scene_paths = [
    constants.SCENES_PATH+'/m1.png',
    constants.SCENES_PATH+'/m2.png',
    constants.SCENES_PATH+'/m3.png',
    constants.SCENES_PATH+'/m4.png',
    constants.SCENES_PATH+'/m5.png'
]

product_paths = [
    constants.MODELS_PATH + '/0.jpg',
    constants.MODELS_PATH +'/1.jpg',
    constants.MODELS_PATH +'/11.jpg',
    constants.MODELS_PATH +'/19.jpg',
    constants.MODELS_PATH +'/24.jpg',
    constants.MODELS_PATH +'/26.jpg',
    constants.MODELS_PATH +'/25.jpg'
]

# READ PARAMS
args =getParams()

results = find_instances(scene_paths, product_paths, args.threshold, args.minMatches,args.cellSize,args.valuesFromMax)

for result in results:
    print(f'scene:{result.scene_name}')
    for model in result.model_instances:
        print(f'    model :{model.model._model_name}')
        print(f'    number of instances found :{model.n_instances}')
        #print(f'    centroids :{model.centroids}')
    print('---------------------------------')
    printSceneAnalisys(result)







