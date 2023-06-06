'''transformation.py
Perform projections, translations, rotations, and scaling operations on Numpy ndarray data.
Hesed Guwn
CS 252 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
import palettable
import analysis
import data

class Transformation(analysis.Analysis):

    def __init__(self, orig_dataset, data=None):
        '''Constructor for a Transformation object

        Parameters:
        -----------
        orig_dataset: Data object. shape=(N, num_vars).
            Contains the original dataset (only containing all the numeric variables,
            `num_vars` in total).
        data: Data object (or None). shape=(N, num_proj_vars).
            Contains all the data samples as the original, but ONLY A SUBSET of the variables.
            (`num_proj_vars` in total). `num_proj_vars` <= `num_vars`

        TODO:
        - Pass `data` to the superclass constructor.
        - Create an instance variable for `orig_dataset`.
        '''
        super().__init__(data)
        self.orig_dataset = orig_dataset

    def project(self, headers):
        '''Project the original dataset onto the list of data variables specified by `headers`,
        i.e. select a subset of the variables from the original dataset.
        In other words, your goal is to populate the instance variable `self.data`.

        Parameters:
        -----------
        headers: Python list of str. len(headers) = `num_proj_vars`, usually 1-3 (inclusive), but
            there could be more.
            A list of headers (strings) specifying the feature to be projected onto each axis.
            For example: if headers = ['hi', 'there', 'cs251'], then the data variables
                'hi' becomes the 'x' variable,
                'there' becomes the 'y' variable,
                'cs251' becomes the 'z' variable.
            The length of the list matches the number of dimensions onto which the dataset is
            projected — having 'y' and 'z' variables is optional.

        TODO:
        - Create a new `Data` object that you assign to `self.data` (project data onto the `headers`
        variables). Determine and fill in 'valid' values for all the `Data` constructor
        keyword arguments (except you dont need `filepath` because it is not relevant here).
        '''
        variable_Data = self.orig_dataset.select_data(headers)
        
        header_Columns = {}
        
        for header in headers:
            header_Columns[header] = headers.index(header)
            
        self.data = data.Data(headers = headers, data = variable_Data, header2col = header_Columns)

    def get_data_homogeneous(self):
        '''Helper method to get a version of the projected data array with an added homogeneous
        coordinate. Useful for homogeneous transformations.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected data array with an added 'fake variable'
        column of ones on the right-hand side.
            For example: If we have the data SAMPLE (just one row) in the projected data array:
            [3.3, 5.0, 2.0], this sample would become [3.3, 5.0, 2.0, 1] in the returned array.

        NOTE:
        - Do NOT update self.data with the homogenous coordinate.
        '''
        original_Data = self.data.get_all_data()
        
        coords = np.ones([original_Data.shape[0],1])
        
        homogeneous_data = np.hstack([original_Data, coords])      
        
        return homogeneous_data
         

    def translation_matrix(self, magnitudes):
        ''' Make an M-dimensional homogeneous transformation matrix for translation,
        where M is the number of features in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these
            amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The transformation matrix.

        NOTE: This method just creates the translation matrix. It does NOT actually PERFORM the
        translation!
        '''
        matrix = np.eye(len(magnitudes) + 1)
        
        matrix[:-1, -1] = magnitudes
        
        return matrix

    def scale_matrix(self, magnitudes):
        '''Make an M-dimensional homogeneous scaling matrix for scaling, where M is the number of
        variables in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The scaling matrix.

        NOTE: This method just creates the scaling matrix. It does NOT actually PERFORM the scaling!
        '''
        matrix = np.eye(len(magnitudes) + 1)
        
        for i in range(len(magnitudes)):
            
            matrix[i][i] = magnitudes[i]
            
        return matrix

    def translate(self, magnitudes):
        '''Translates the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The translated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to translate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''
        geneous_data = self.get_data_homogeneous()
        
        matrix = self.translation_matrix(magnitudes) @ geneous_data.T
        
        matrix = matrix.T[:, :-1]
        
        self.data = data.Data(data = matrix)
        
        return self.data.get_all_data()

    def scale(self, magnitudes):
        '''Scales the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The scaled data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to scale the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        geneous_data = self.get_data_homogeneous()
        
        matrix = self.scale_matrix(magnitudes) @ geneous_data.T
        
        matrix = matrix.T[:, :-1]
        
        self.data = data.Data(data = matrix)
        
        return self.data.get_all_data()

    def transform(self, C):
        '''Transforms the PROJECTED dataset by applying the homogeneous transformation matrix `C`.

        Parameters:
        -----------
        C: ndarray. shape=(num_proj_vars+1, num_proj_vars+1).
            A homogeneous transformation matrix.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The projected dataset after it has been transformed by `C`

        TODO:
        - Use matrix multiplication to apply the compound transformation matix `C` to the projected
        dataset.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''
        geneous_data = self.get_data_homogeneous()
        
        matrix = C @ geneous_data.T
        
        matrix = matrix.T[:,:-1]
        
        self.data = data.Data(data = matrix, headers = self.data.headers, header2col = self.data.header2col)

        return self.data.get_all_data()
        

    def normalize_together(self):
        '''Normalize all variables in the projected dataset together by translating the global minimum
        (across all variables) to zero and scaling the global range (across all variables) to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''
        transform_numbers = []
        
        scale_numbers = []
        
        for i in range(len(self.data.get_headers())):
            
            transform_numbers.append(-1*(self.data.get_all_data().min()))
            
            scale_numbers.append(1/((self.data.get_all_data().max()) - self.data.get_all_data().min()))
            
        scale_matrix = self.scale_matrix(scale_numbers)
        
        translation_matrix = self.translation_matrix(transform_numbers)
        
        norm = scale_matrix @ translation_matrix
        
        return self.transform(norm)

    def normalize_separately(self):
        '''Normalize each variable separately by translating its local minimum to zero and scaling
        its local range to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''
        transform_numbers = []
        
        scale_numbers = []
        
        gmin = self.min(self.data.get_headers()) 
        
        gmax = self.max(self.data.get_headers())
        
        for i in range(len(self.data.get_headers())):
            
            transform_numbers.append(-1*(gmin[i]))
            
            scale_numbers.append( 1 / (gmax[i] - gmin[i])  )
            
        scale_matrix = self.scale_matrix(scale_numbers)
        
        translation_matrix = self.translation_matrix(transform_numbers)
        
        norm = scale_matrix @ translation_matrix
        
        return self.transform(norm)

    def rotation_matrix_3d(self, header, degrees):
        '''Make an 3-D homogeneous rotation matrix for rotating the projected data
        about the ONE axis/variable `header`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(4, 4). The 3D rotation matrix with homogenous coordinate.

        NOTE: This method just creates the rotation matrix. It does NOT actually PERFORM the rotation!
        '''
        ident = np.eye(4)
        
        index = self.data.header2col[header]
            
        degrees = np.deg2rad(degrees)
        
        if index == 0:
            
            arr1 = [1, 0, 0, 0], [0,np.cos(degrees),-(np.sin(degrees)),0], [0, (np.sin(degrees)), np.cos(degrees), 0], [0,0,0,1]
            
            r_trans = np.array(arr1)
            
            return ident @ r_trans
    
        if index == 1:
            
            arr1 = [np.cos(degrees), 0, np.sin(degrees), 0], [0,1,0,0], [-(np.sin(degrees)), 0, np.cos(degrees), 0], [0,0,0,1]
            
            r_trans = np.array(arr1)
            
            return ident @ r_trans
        
        if index == 2:
            
            arr1 = [np.cos(degrees),-(np.sin(degrees)),0, 0], [(np.sin(degrees)), np.cos(degrees), 0, 0], [0,0,1,0], [0,0,0,1]
            
            r_trans = np.array(arr1)
            
            return ident @ r_trans

    def rotate_3d(self, header, degrees):
        '''Rotates the projected data about the variable `header` by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to rotate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        geneous_data = self.get_data_homogeneous()
        
        matrix = self.rotation_matrix_3d(header, degrees) @ geneous_data.T
        
        matrix = matrix.T[:, :-1]
        
        self.data = data.Data(data = matrix, headers = self.data.headers, header2col= self.data.header2col)
        
        return self.data.get_all_data()

    def scatter_color(self, ind_var, dep_var, c_var, title=None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
            NOTE: Use a ColorBrewer color palette (e.g. from the `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''
        color_map = palettable.colorbrewer.sequential.YlGnBu_7
        
        scatter = plt.scatter(self.orig_dataset.select_data([ind_var]), self.orig_dataset.select_data([dep_var]), c = self.orig_dataset.select_data([c_var]), cmap=color_map.mpl_colormap)
        
        plt.title(title)
        
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
       
        plt.colorbar(scatter, label = c_var)
        
        plt.show()