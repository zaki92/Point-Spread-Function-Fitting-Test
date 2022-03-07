# -*- coding: utf-8 -*-
"""
Given a tab delimited text file having values of x, y ,z and f(x,y,z), 
hypothesize that the f(x,y,z) ~ 3D gaussian. Find of best parameters of the 3D 
gaussian using a steepest descent algorithm - this is a manual check of the 
output of a new image analysis program for a new lattice light field microscope

Input: 
    Tab delimited file, having values of x, y, z, and f(x,y,z)
    
Output: 
    3D Gaussian Parameters, comparitive plots. 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
import time

def generate_test_data(amp, sig_x, sig_y, sig_z, x0, y0, z0, const):
    
    """Generates numpy arrays of test data corresponding to a 
       symmetric gaussian"""
    
    x_range = np.arange(0,64, 1)
    y_range = np.arange(0,64, 1)
    z_range = np.arange(0,100, 1)
    
    x, y, z = np.meshgrid(x_range, y_range, z_range)
    
    x_val = x.flatten()
    y_val = y.flatten()
    z_val = z.flatten()
    
    x_term = (x_val - x0)**2 / (2*sig_x**2)
    y_term = (y_val - y0)**2 / (2*sig_y**2)
    z_term = (z_val - z0)**2 / (2*sig_z**2) 

    f_val = amp*np.exp(- x_term - y_term - z_term) + const
    
    return x_val, y_val, z_val ,f_val

def mean_sq_error(parameter_array, x_val, y_val, z_val, f_val):

    """The parameter array is the EM's best guess of the 3D parameters, 
       a MSE is calculated and passed to the scipy.optimize function for
       steepest descent minimization"""
       
    amp = parameter_array[0]
    sigma_x = parameter_array[1]
    sigma_y = parameter_array[2]
    sigma_z = parameter_array[3]
    x0 = parameter_array[4]
    y0 = parameter_array[5]
    z0 = parameter_array[6]
    cons = parameter_array[7]
    
    x_term = (x_val - x0)**2 / (2*sigma_x**2)
    y_term = (y_val - y0)**2 / (2*sigma_y**2)
    z_term = (z_val - z0)**2 / (2*sigma_z**2) 

    hypothesis = amp*np.exp(- x_term - y_term - z_term) + cons

    m = len(f_val)

    error = (1/(2*m)) * np.sum((hypothesis - f_val)**2)
    
    # uncomment to check if mse error values decrease as minimization happens
    #print(error)

    return error


start_time = time.time()


#it is possible to use numpy to read the data directly too 
df = pd.read_table('560_PSF.txt') #read the file into a dataframe 

x_val = np.asarray(df['x']) 
y_val = np.asarray(df['y'])
z_val = np.asarray(df['z'])
f_val = np.asarray(df['f'])


parameter_array_guess = np.array([1000,20,20,20,30,30,50,125])

"""Check res.status and res.message for proof of successful minimization"""

options= {'maxiter': 10000,'disp':True}

res = optimize.minimize(mean_sq_error,
                                parameter_array_guess,
                                (x_val,y_val,z_val,f_val),
                                jac=False,
                                method='L-BFGS-B',
                                options = options)

#outputs the result, and message success/condition
print(res)

#results for 488nm (18715, 2.04, 2.75, 3.81,29.1, 34.4,51.9,125)
#results for 560nm (5440, 2.01, 2.64, 4.2, 28.9, 34.8, 49.4)
"""The max amp is 5440 (arbitrary intensity units), sigmas are 2.01, 2.64, 
   4.2 (in voxel units), and the PSF is centered at (34.4, 51.9, 125)"""


x_fit, y_fit, z_fit, f_fit = generate_test_data(5439, 2.04, 2.64, 4.20,28.9, 34.8,49.4,114)


#use this snippet to plot the gaussian test data along the x axis 
bool_mask_1 = y_val == 35 #center y coordinate
bool_mask_2 = z_val == 49 #center x coordinate
bool_mask = bool_mask_1 & bool_mask_2

bool_mask_fit_1 = y_fit == 35 #center y coordinate
bool_mask_fit_2 = z_fit == 49 #center x coordinate
bool_mask_fit = bool_mask_fit_1 & bool_mask_fit_2

plt.plot(x_val[bool_mask],f_val[bool_mask], label = 'PSF')
plt.plot(x_fit[bool_mask_fit],f_fit[bool_mask_fit], color = 'red',label = 'Gaussian Fit')
plt.xlabel('Voxel X-Coordinate')
plt.ylabel('Intensity Value')
plt.legend()

print('Total Time:', time.time() - start_time)


"""
#x_val, y_val, z_val, f_val = generate_test_data(100, 35, 24, 25, 25, 65, 55, 0)

#use this snippet to print unique counts in each val
unique, counts = np.unique(f_val, return_counts=True)

print (np.asarray((unique, counts)).T)
"""





