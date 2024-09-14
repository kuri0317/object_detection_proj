import numpy as np
import dataclasses
# accumulator is an image (numpy array with extra dimension for number of votes
# votes are casted in the exact coordinates,
# the getMax scans the image and counts the votes in a subsection of the image of cell_size dim
# accumulator as a bidimensional array of the same size of the image, initialized with zeroes, vote casted in the exact coordinates, and than aggregates with the

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
        try:
            self.acc_array[int(centroid[1]/self.cell_size),int(centroid[0]/self.cell_size)]+=1
            #self.acc_array[int(centroid[1]/self.cell_size),int(centroid[0]/self.cell_size),1].append(keypoint)
            self.voters_array[int(centroid[1]/self.cell_size),int(centroid[0]/self.cell_size)].append(keypoint)
        except IndexError:
            print(f"invalid index {int(centroid[1])}:{int(centroid[0])}")

    def  getMax(self):
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
        coordinates=np.where(self.acc_array==np.max(self.acc_array))
        i = 0
        while i<len(coordinates[0]):
            if len(self.voters_array[coordinates[0][i]][coordinates[1][i]]) >0:
                print(f"asdrubale{self.voters_array[coordinates[0][i]][coordinates[1][i]]}")
            centroids.append([coordinates[1][i]*self.cell_size,coordinates[0][i]*self.cell_size,self.voters_array[coordinates[0][i]][coordinates[1][i]]])
            i+=1

        max_value= np.max(self.acc_array)
        return max_value,centroids
