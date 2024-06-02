import argparse
import lib.constants as constants
# parameter parser function
def getParams():
    '''
    Parse command line arguments

    '''
    parser = argparse.ArgumentParser(prog='object_detection_project',description='box detection project',epilog='credits carnivuth,kuri')
    parser.add_argument('-t','--threshold',default=constants.THRESHOLD,help='thrashold for ratio test',type=float)
    parser.add_argument('-m','--minMatches',default=constants.MIN_MATCHES,help='minimum number of matches for detecting the model in the target image',type=int)
    return parser.parse_args()
