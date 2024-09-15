import numpy as np
import dataclasses
from lib import constants as c

@dataclasses.dataclass
class Accumulator:
    '''
    Class representation of the ght accumulator
    '''
    cell_size: int
    acc_array: np.ndarray
    voters_array: np.ndarray

    def __init__(self,image:np.ndarray,cell_size:int):
        self.cell_size = cell_size
        self.acc_array= np.zeros((image.shape[0]//self.cell_size,image.shape[1]//self.cell_size))
        self.voters_array= np.empty((image.shape[0]//self.cell_size,image.shape[1]//self.cell_size),dtype='O')
        for i in range(self.voters_array.shape[0]):
            for j in range(self.voters_array.shape[1]):
                self.voters_array[i][j]=[]


    def castVote(self,centroid:np.ndarray,keypoint:np.ndarray):
        """
        Cast a vote in the accumulator given a computed centroid

        Parameters:
        -----------
            centroid: np.ndarray
                numpy array with the centroid coordinates

        """
        # cast vote by resizing the coordinates based on the cell size, try/except needed for out of bound votes
        try:
            self.acc_array[int(centroid[1]//self.cell_size),int(centroid[0]//self.cell_size)]+=1
            self.voters_array[int(centroid[1]//self.cell_size),int(centroid[0]//self.cell_size)].append(keypoint)
        except IndexError:
            print(f"invalid index {int(centroid[1])}:{int(centroid[0])}")

    def  getMax(self,values_from_max=c.VALUES_FROM_MAX):
        """
        Get centroid with the max number of votes

        Returns:
        -----------
            n_votes: int
                maximum number of votes
            max: AccumulatorCell
                the most voted cell in the array

        """
        centroids=[]

        # get max of accumulator and values under max as set in constants
        max_value= np.max(self.acc_array)
        coordinates=np.where(self.acc_array==max_value)
        for i in range(values_from_max):
            coordinates+=np.where(self.acc_array==(max_value-i))

        for i in range(len(coordinates[0])):
            centroids.append([coordinates[1][i]*self.cell_size,coordinates[0][i]*self.cell_size,self.voters_array[coordinates[0][i]][coordinates[1][i]]])

        return max_value,centroids
