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

# Plot array containing the chosen phonemes

# Create a figure and a subplot
fig, ax1 = plt.subplots()

title_string = 'Phoneme 1 & 2'
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 & 2)
plot_data(X=X_phonemes_1_2, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_phonemes_1_2.png')
plt.savefig(plot_filename)


#########################################
# Write your code here
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 1
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 2
# Compare these predictions for each sample of the dataset, and calculate the accuracy, and store it in a scalar variable named "accuracy"
#phnoneme 1
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
Z_1 = np.zeros((N_1 + N_2,k)) # shape Nxk

###############################
# run Expectation Maximization algorithm for n_iter iterations
for t in range(n_iter_1):
    print('Iteration {:03}/{:03}'.format(t+1, n_iter_1))

    # Do the E-step
    Z_1 = get_predictions(mu_1, s_1, p_1, X_phonemes_1_2)
    Z_1 = normalize(Z_1, axis=1, norm='l1')

    # Do the M-step:
    for i in range(k):
        mu_1[i,:] = np.matmul(X_phonemes_1_2.transpose(),Z_1[:,i]) / np.sum(Z_1[:,i])
        # We will fit Gaussians with diagonal covariance matrices
        mu_i_1 = mu_1[i,:]
        mu_i_1 = np.expand_dims(mu_i_1, axis=1)
        mu_i_repeated_1 = np.repeat(mu_i_1, N_1 + N_2, axis=1)
        X_minus_mu_1 = (X_phonemes_1_2.transpose() - mu_i_repeated_1)**2
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
Z_2 = np.zeros((N_2 + N_1,k)) # shape Nxk

###############################
# run Expectation Maximization algorithm for n_iter iterations
for t in range(n_iter_2):
    print('Iteration {:03}/{:03}'.format(t+1, n_iter_2))

    # Do the E-step
    Z_2 = get_predictions(mu_2, s_2, p_2, X_phonemes_1_2)
    Z_2 = normalize(Z_2, axis=1, norm='l1')

    # Do the M-step:
    for i in range(k):
        mu_2[i,:] = np.matmul(X_phonemes_1_2.transpose(),Z_2[:,i]) / np.sum(Z_2[:,i])
        # We will fit Gaussians with diagonal covariance matrices
        mu_i_2 = mu_2[i,:]
        mu_i_2 = np.expand_dims(mu_i_2, axis=1)
        mu_i_repeated_2 = np.repeat(mu_i_2, N_2 + N_1, axis=1)
        X_minus_mu_2 = (X_phonemes_1_2.transpose() - mu_i_repeated_2)**2
        res_2 = np.squeeze( np.matmul(X_minus_mu_2, np.expand_dims(Z_2[:,i], axis=1)))/np.sum(Z_2[:,i])
        s_2[i,:,:] = np.diag(res_2)
        p_2[i] = np.mean(Z_2[:,i])

false = 0
right = 0
accout = 0
for i in range(len(Z_1)):
    if Z_1[i][0] + Z_1[i][1] + Z_1[i][2] > Z_2[i][0] + Z_2[i][1] + Z_2[i][2]:
        if i not in Ph_1 and i in Ph_2:
            false += 1
        else:
            right += 1

    elif Z_1[i][0] + Z_1[i][1] + Z_1[i][2] < Z_2[i][0] + Z_2[i][1] + Z_2[i][2]:
        if i not in Ph_2 and i in Ph_1:
            false += 1
        else:
            right += 1
    else:
        right += 1
        accout += 1
accuracy = right / (false + right) * 100
########################################/

print('Accuracy using GMMs with {} components: {:.2f}%'.format(k, accuracy))

################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()