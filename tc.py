import numpy as np
from algo import ValueFunctionWithApproximation

class ValueFunctionWithTile(ValueFunctionWithApproximation):
    def __init__(self,
                 state_low:np.array, #(2d array, position, velocity)
                 state_high:np.array, #(2d array, position, velocity)
                 num_tilings:int,
                 tile_width:np.array): #2d array
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # TODO: implement this method
        self.state_low = state_low
        self.state_high = state_high
        self.num_tilings = num_tilings
        self.tile_width = tile_width  # In test it is a 2d array
     
        # Calculate the number of tiles for each tiling
        #Row/Col number
        self.num_tiles_row = np.ceil((state_high[0] - state_low[0]) / tile_width[0]).astype(int) + 1 #location
        self.num_tiles_col = np.ceil((state_high[1] - state_low[1]) / tile_width[1]).astype(int) + 1 #velocity
        
        # Initialize offsets for each tiling
        self.offsets_row = np.array([(-i / num_tilings) * tile_width[0] for i in range(num_tilings)])
        self.offsets_col = np.array([(-i / num_tilings) * tile_width[1] for i in range(num_tilings)])

        
        # Total number of features (tiles) across all tilings
        self.num_features = num_tilings * self.num_tiles_row *self.num_tiles_col 
        
        # Initialize weights for linear approximation
        self.weights = np.zeros(self.num_features)  # 1D array

    def __call__(self,s):
        """
        Returns the value of the given state s using the current weights.
        """
        tile_indices = self._get_active_tiles(s) #Get a list/array contains the index of tile in each tiling flat

        return np.sum(self.weights[tile_indices])

    def update(self,alpha,G,s_tau):
        """
        Updates the weights based on the TD error.
        """
        tile_indices = self._get_active_tiles(s_tau)
        delta = alpha * (G - self.__call__(s_tau))
        self.weights[tile_indices] += delta

    def _get_active_tiles(self, s):
            """
            Helper function to compute the active tiles (indices) for a given state s.
            """
            tile_indices = []
            
            for i in range(self.num_tilings):
                # Compute the ind for s in each tiling flat
                cell_ind_row =  ((s[0] - (self.state_low[0] + self.offsets_row[i] )) // self.tile_width[0]).astype(int)
                cell_ind_col =  ((s[1] - (self.state_low[1] + self.offsets_col[i] )) // self.tile_width[1]).astype(int)
                ind_in_single_flat = np.ravel_multi_index(np.array([cell_ind_row,cell_ind_col]), (self.num_tiles_row,self.num_tiles_col))
                tile_ind_in_all_tiling_flat = i *self.num_tiles_row * self.num_tiles_col + ind_in_single_flat
                tile_indices.append(tile_ind_in_all_tiling_flat)

            return np.array(tile_indices)