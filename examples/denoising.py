def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

import numpy as np
from scipy import ndimage as img
from scipy import io as sio
import matplotlib.pyplot as plt
import pyshearlab

tic()
print("--SLExampleImageDenoising")
print("loading image...")

sigma = 30
scales = 3
thresholdingFactor = 3

# load data
X = img.imread("barbara.jpg")[::4, ::4]
X = X.astype(float)

# add noise
Xnoisy = X + sigma*np.random.randn(X.shape[0], X.shape[1])
toc()

tic()
print("generating shearlet system...")
## create shearlets
shearletSystem = pyshearlab.SLgetShearletSystem2D(0,X.shape[0], X.shape[1], scales)

toc()
tic()
print("decomposition, thresholding and reconstruction...")

# decomposition
coeffs = pyshearlab.SLsheardec2D(Xnoisy, shearletSystem)

# thresholding
oldCoeffs = coeffs.copy()
weights = np.ones(coeffs.shape)

for j in range(len(shearletSystem["RMS"])):
    weights[:,:,j] = shearletSystem["RMS"][j]*np.ones((X.shape[0], X.shape[1]))
    
coeffs = np.real(coeffs)
zero_indices = np.abs(coeffs) / (thresholdingFactor * weights * sigma) < 1
coeffs[zero_indices] = 0

# reconstruction
Xrec = pyshearlab.SLshearrec2D(coeffs, shearletSystem)
toc()
PSNR = pyshearlab.SLcomputePSNR(X,Xrec)
print("PSNR: " + str(PSNR))
#sio.savemat("PyShearLab_DenoisingExample.mat", {"weights": weights, "XPyNoisy": Xnoisy,
# "XPyDenoised": Xrec, "PyPSNR": PSNR, "coeffThrPy": coeffs, "oldCoeffs": oldCoeffs})
plt.gray()
plt.imshow(Xrec)
plt.colorbar()
plt.show()
