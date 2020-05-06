# Parameters for the Hard Sphere fitting
## k range for the fitting
fitmin = 0
fitmax = 0.6

# Input/Output
## The input P(k) to be fitted, with the first 2 columns being (k, P(k))
input_data = "input/input_data.dat"
## Directory for outputs.
output_dir = "output/"

# Covariance matrix
## If `compute_cov` = True, then read `input_mocks` for a list of mock files,
##   and then compute the covariance matrix using the mocks.
## If `compute_cov` = True and `save_cov` = True, then write the pre-processed
##   covariance matrix to file `cov_file`.
## If `compute_cov` = False, then read the covariance matrix from `cov_file`.
compute_cov     = True
save_cov        = True
input_mocks     = "input/list_mocks.dat"
cov_file        = "input/cov.dat"

# Theoretical model
model_kmin      = 1e-4
model_kmax      = 1e4
model_Nk        = 1e6
