# imaging-methods
Some diverse python codes I have developed and use to analyze imaging data at the boundary of tokamaks. In particular,
I have applied this to APD, Phantom and synthetic blob data. The main method is based on two-dimensional conditional averaging
(2DCA), and other methods work on the output data from 2DCA. The package includes the following files:
- cond_av: Code for two-dimensional conditional averaging
- contours: Blob tracking based on contouring
- data_preprocessing: Method to handle and preprocess experimental data
- discharge: Dataclasses definitions for shots and analysis results
- duration_time_estimation: Method to estimate duration time by fitting power spectral density or auto-correlation function
- parameter_estimation: Other estimation methods applied to the outcome of 2DCA.
- show_data: Diverse plotting functions
- utils: Other functions

Top level files have been used to analyze different shots and synthetic data generated with blobmodel.