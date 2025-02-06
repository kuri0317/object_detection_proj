import numpy as np
from scipy.ndimage import minimum_filter,maximum_filter, label
import dataclasses

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
            #print(f"invalid index {int(centroid[1])}:{int(centroid[0])}")
            pass

    def  getMax(self,neighborhood_size):
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

        data_max = maximum_filter(self.acc_array, neighborhood_size)
        maxima = (self.acc_array == data_max)
        maxima_thresholded = self.acc_array > 6
        labeled, _ = label(maxima_thresholded & maxima)
        max_voted_cells = np.column_stack(np.where(labeled > 0))

        for centroid in max_voted_cells:

            centroids.append(
                [
                    centroid[1]*self.cell_size,
                    centroid[0]*self.cell_size,
                    self.voters_array[centroid[0],centroid[1]]
                ]
            )

        return 0,centroids
