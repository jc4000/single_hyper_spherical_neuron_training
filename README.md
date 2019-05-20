# single_hyper_spherical_neuron_training
For training of a hyper-spherical neuron:
------------------------------
The description of the theory:
-------------------------------
This program is implementation of a hyper-sphere neuron training. 
The output of the neuron is: S(z_i) = 1 / (1 + exp(Alpha*z_i)), where 
z_i = (c_0 - x_i0)**2 + ...+(c_n - x_in)**2 - R**2 is the square of the distance of input x_i from the surface of the 
hyper-sphere, c_j is a component of the center of the hyper-sphere, R is the radius of the hyper-sphere and 
Alpha > 0 is a scaling factor.

The activation S(z) is mirror reflection of classic sigmoid. S(z) approaches 0 if z approaches infinity and S(z) approaches
1 if z approaches negative infinity.
---------------------------------------------
---------------------------------
Instructions to execute
---------------------------------
Save the file, binary_nand.csv in the same directory where this program is saved

To train with configuration other than the defaults:
Execute the command 'python crclr_single_nrn_entrpy_err.py binary_nand.csv network_config.json'

binary_nand.csv is the file with training data
network_config.json is the file with training configuration details

OR

To train with default configurations:
Execute the command 'python crclr_single_nrn_entrpy_err.py binary_nand.csv'

binary_nand.csv is the file with training data

----------------------------------
Expected Output
----------------------------------
* The convergence status in terms of RMS error and iteration count are displayed on the command shell.
* A file containing the convergence values and the final parameter values are also saved. The file name can be 
  altered in the code at line number 563. The file name starts with 'progress_status'.
