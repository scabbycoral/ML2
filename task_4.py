import numpy as np
import os
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *
import random
from sklearn.preprocessing import normalize
from get_predictions import *
from plot_gaussians import *

# File that contains the data
data_npy_file = 'data/PB_data.npy'

# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)

# Make a folder to save the figures
figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']
# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
#########################################
# Write your code here
# Store f1 in the first column of X_full, and f2 in the second column of X_full
for i in range(len(f1)):
    X_full[i] = [f1[i], f2[i]]
########################################/
X_full = X_full.astype(np.float32)

# number of GMM components
k = 3

#########################################
# Write your code here

# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and samples that belong to phoneme 2.
# The shape of X_phonemes_1_2 will be two-dimensional. Each row will represent a sample of the dataset, and each column will represent a feature (e.g. f1 or f2)
# Fill X_phonemes_1_2 with the samples of X_full that belong to the chosen phonemes
# To fill X_phonemes_1_2, you can leverage the phoneme_id array, that contains the ID of each sample of X_full

# X_phonemes_1_2 = ...
X_phonemes_1_2 = np.zeros((np.sum(phoneme_id==1) + np.sum(phoneme_id==2), 2))
i = 0
for id in range(len(phoneme_id)):
    if phoneme_id[id] == 1 or phoneme_id[id] == 2:
        X_phonemes_1_2[i] = X_full[id]
        i += 1

Ph_1 = []
Ph_2 = []
X_phonemes_1 = np.zeros((np.sum(phoneme_id==1), 2))
i = 0
for id in range(len(phoneme_id)):
    if phoneme_id[id] == 1:
        X_phonemes_1[i] = X_full[id]
        Ph_1.append(id)
        i += 1

X_phonemes_2 = np.zeros((np.sum(phoneme_id==2), 2))
i = 0
for id in range(len(phoneme_id)):
    if phoneme_id[id] == 2:
        X_phonemes_2[i] = X_full[id]
        Ph_2.append(id)
        i += 1
########################################/

# as dataset X, we will use only the samples of phoneme 1 and 2
#X = X_phonemes_1_2.copy()
X = np.vstack((X_phonemes_1, X_phonemes_2))
########################################
min_f1 = int(np.min(X[:,0]))
max_f1 = int(np.max(X[:,0]))
min_f2 = int(np.min(X[:,1]))
max_f2 = int(np.max(X[:,1]))
N_f1 = max_f1 - min_f1
N_f2 = max_f2 - min_f2
print('f1 range: {}-{} | {} points'.format(min_f1, max_f1, N_f1))
print('f2 range: {}-{} | {} points'.format(min_f2, max_f2, N_f2))

#########################################
# Write your code here

# Create a custom grid of shape N_f1 x N_f2
# The grid will span all the values of (f1, f2) pairs, between [min_f1, max_f1] on f1 axis, and between [min_f2, max_f2] on f2 axis
# Then, classify each point [i.e., each (f1, f2) pair] of that grid, to either phoneme 1, or phoneme 2, using the two trained GMMs
# Do predictions, using GMM trained on phoneme 1, on custom grid
# Do predictions, using GMM trained on phoneme 2, on custom grid
# Compare these predictions, to classify each point of the grid
# Store these prediction in a 2D numpy array named "M", of shape N_f2 x N_f1 (the first dimension is f2 so that we keep f2 in the vertical axis of the plot)
# M should contain "0.0" in the points that belong to phoneme 1 and "1.0" in the points that belong to phoneme 2
########################################/
X_1 = X_phonemes_1.copy()
# get number of samples
N_1 = X_1.shape[0]
# get dimensionality of our dataset
D_1 = X_1.shape[1]

# common practice : GMM weights initially set as 1/k
p_1 = np.ones((k))/k
# GMM means are picked randomly from data samples
random_indices_1 = np.floor(N_1*np.random.rand((k)))
random_indices_1 = random_indices_1.astype(int)
mu_1 = X_1[random_indices_1,:] # shape kxD
# covariance matrices
s_1 = np.zeros((k,D_1,D_1)) # shape kxDxD
# number of iterations for the EM algorithm
n_iter_1 = 100

#phnoneme 2
X_2 = X_phonemes_2.copy()
# get number of samples
N_2 = X_2.shape[0]
# get dimensionality of our dataset
D_2 = X_2.shape[1]

# common practice : GMM weights initially set as 1/k
p_2 = np.ones((k))/k
# GMM means are picked randomly from data samples
random_indices_2 = np.floor(N_1*np.random.rand((k)))
random_indices_2 = random_indices_2.astype(int)
mu_2 = X_2[random_indices_2,:] # shape kxD
# covariance matrices
s_2 = np.zeros((k,D_2,D_2)) # shape kxDxD
# number of iterations for the EM algorithm
n_iter_2 = 100

# initialize covariances
for i in range(k):
    cov_matrix_1 = np.cov(X_1.transpose())
    # initially set to fraction of data covariance
    s_1[i,:,:] = cov_matrix_1/k

# Initialize array Z that will get the predictions of each Gaussian on each sample
Z_1 = np.zeros((N_1,k)) # shape Nxk

###############################
# run Expectation Maximization algorithm for n_iter iterations
for t in range(n_iter_1):
    print('Iteration {:03}/{:03}'.format(t+1, n_iter_1))

    # Do the E-step
    Z_1 = get_predictions(mu_1, s_1, p_1, X_phonemes_1)
    Z_1 = normalize(Z_1, axis=1, norm='l1')

    # Do the M-step:
    for i in range(k):
        mu_1[i,:] = np.matmul(X_phonemes_1.transpose(),Z_1[:,i]) / np.sum(Z_1[:,i])
        # We will fit Gaussians with diagonal covariance matrices
        mu_i_1 = mu_1[i,:]
        mu_i_1 = np.expand_dims(mu_i_1, axis=1)
        mu_i_repeated_1 = np.repeat(mu_i_1, N_1, axis=1)
        X_minus_mu_1 = (X_phonemes_1.transpose() - mu_i_repeated_1)**2
        res_1 = np.squeeze( np.matmul(X_minus_mu_1, np.expand_dims(Z_1[:,i], axis=1)))/np.sum(Z_1[:,i])
        s_1[i,:,:] = np.diag(res_1)
        p_1[i] = np.mean(Z_1[:,i])


#phnoneme 2


# initialize covariances
for i in range(k):
    cov_matrix_2 = np.cov(X_2.transpose())
    # initially set to fraction of data covariance
    s_2[i,:,:] = cov_matrix_2/k

# Initialize array Z that will get the predictions of each Gaussian on each sample
Z_2 = np.zeros((N_2,k)) # shape Nxk

###############################
# run Expectation Maximization algorithm for n_iter iterations
for t in range(n_iter_2):
    print('Iteration {:03}/{:03}'.format(t+1, n_iter_2))

    # Do the E-step
    Z_2 = get_predictions(mu_2, s_2, p_2, X_phonemes_2)
    Z_2 = normalize(Z_2, axis=1, norm='l1')

    # Do the M-step:
    for i in range(k):
        mu_2[i,:] = np.matmul(X_phonemes_2.transpose(),Z_2[:,i]) / np.sum(Z_2[:,i])
        # We will fit Gaussians with diagonal covariance matrices
        mu_i_2 = mu_2[i,:]
        mu_i_2 = np.expand_dims(mu_i_2, axis=1)
        mu_i_repeated_2 = np.repeat(mu_i_2, N_2, axis=1)
        X_minus_mu_2 = (X_phonemes_2.transpose() - mu_i_repeated_2)**2
        res_2 = np.squeeze( np.matmul(X_minus_mu_2, np.expand_dims(Z_2[:,i], axis=1)))/np.sum(Z_2[:,i])
        s_2[i,:,:] = np.diag(res_2)
        p_2[i] = np.mean(Z_2[:,i])

f1 = np.arange(min_f1, max_f1)
f2 = np.arange(min_f2, max_f2)
f3 = np.tile(f1, len(f2))
f4 = np.repeat(f2, len(f1))
grid = np.transpose([np.tile(f1, len(f2)), np.repeat(f2, len(f1))])
grid_1_2 = grid.copy()
result1 = np.zeros((1, k))
result2 = np.zeros((1, k))
Flag = True
#max_items = np.random.choice(range(grid_1_2.shape[0]), size=3000, replace=False)
while len(grid_1_2):
    grid_copy = grid_1_2[:1000]
    P1 = get_predictions(mu_1, s_1, p_1, grid_copy)
    P2 = get_predictions(mu_2, s_2, p_2, grid_copy)
    result1 = np.vstack((result1, P1.copy()))
    result2 = np.vstack((result2, P2.copy()))
    grid_1_2 = grid_1_2[1000 - len(grid_1_2):]
    if not Flag:
        break
    if len(grid_1_2) == 1000:
        Flag = False
    #print(len(grid_1_2))

result1 = result1[1 - len(result1):]
result2 = result2[1 - len(result2):]

MMM = np.zeros((N_f1, N_f2))
MM = MMM.copy()
num1 = 0
num2 = 0
num3 = 0

for i in range(len(result1)):
    if result1[i][0] + result1[i][1] + result1[i][2] > result2[i][0] + result2[i][1] + result2[i][2]:
        MM[grid[i][0] - min_f1][grid[i][1] - min_f2] = 1
    elif result1[i][0] + result1[i][1] + result1[i][2] < result2[i][0] + result2[i][1] + result2[i][2]:
        MM[grid[i][0] - min_f1][grid[i][1] - min_f2] = 2


print(num1)
print(num2)
print(num3)
M = MM.transpose()

M[0][0] = 1
M[1][0] = 2
################################################
# Visualize predictions on custom grid

# Create a figure
#fig = plt.figure()
fig, ax = plt.subplots()

# use aspect='auto' (default is 'equal'), to force the plotted image to be square, when dimensions are unequal
plt.imshow(M, aspect='auto')
# set label of x axis
ax.set_xlabel('f1')
# set label of y axis
ax.set_ylabel('f2')

# set limits of axes
plt.xlim((0, N_f1))
plt.ylim((0, N_f2))

# set range and strings of ticks on axes
x_range = np.arange(0, N_f1, step=50)
x_strings = [str(x+min_f1) for x in x_range]
plt.xticks(x_range, x_strings)
y_range = np.arange(0, N_f2, step=200)
y_strings = [str(y+min_f2) for y in y_range]
plt.yticks(y_range, y_strings)

# set title of figure
title_string = 'Predictions on custom grid'
plt.title(title_string)

# add a colorbar
plt.colorbar()

N_samples = int(X.shape[0]/2)
plt.scatter(X[:N_samples, 0] - min_f1, X[:N_samples, 1] - min_f2, marker='.', color='red', label='Phoneme 1')
plt.scatter(X[N_samples:, 0] - min_f1, X[N_samples:, 1] - min_f2, marker='.', color='green', label='Phoneme 2')

# add legend to the subplot
plt.legend()

# save the plotted points of the chosen phoneme, as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'GMM_predictions_on_grid.png')
plt.savefig(plot_filename)

################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()