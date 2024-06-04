import lib.constants as constants
from lib.ght_sift import  find_instances
from lib.cli import getParams
from lib.print_utils import printCentroids

# main
scene_paths = [constants.SCENES_PATH+'/m1.png', constants.SCENES_PATH+'/m2.png', constants.SCENES_PATH+'/m3.png', constants.SCENES_PATH+'/m4.png', constants.SCENES_PATH+'/m5.png']
product_paths = [constants.MODELS_PATH + '/0.jpg', constants.MODELS_PATH +'/1.jpg', constants.MODELS_PATH +'/11.jpg', constants.MODELS_PATH +'/19.jpg', constants.MODELS_PATH +'/24.jpg', constants.MODELS_PATH +'/26.jpg', constants.MODELS_PATH +'/25.jpg']

args = getParams()
results = find_instances(scene_paths, product_paths, args.threshold, args.minMatches)

for result in results:
    print(f'scene:{result.scene_name}')
    for model in result.model_instances:
        print(f'    model :{model.model._model_name}')
        print(f'    number of instances found :{model.n_instances}')
        print(f'    centroids :{model.centroids}')
    print('---------------------------------')
    printCentroids(result)
