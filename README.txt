This is a two dimensional version of the MATLAB ShearLab3D toolbox
translated to Python which has all fundamental functionality for the
two dimensional case. For the original MATLAB toolbox we refer to
http://www3.math.tu-berlin.de/numerik/www.shearlab.org/software.

To install and use the toolbox, please refer to the pySLExampleDenoising.py
file which demonstrates how to compute the forward and the backward
shearlet transform using PyShearLab. 

For convenience and to compare the performence to the MATLAB version, 
a MAT file is created which can be used in MATLAB for comparison benchmarks. 
Since the resulting file would be quiet large (about 300MB), to actually obtain
the file, you thus have to uncomment line 69 in pySLExampleDenoising.py.

Copyright information can be found in copyright.txt. This toolbox
is a Python rewrite of ShearLab3Dv11 by Rafael Reisenhofer.

The methods 
- dfilters
- dmaxflat
- mctrans
- modulate2

were taken from the Nonsubsampled Contourlet Toolbox [1] which can be downloaded 
from http://www.mathworks.de/matlabcentral/fileexchange/10049-nonsubsampled-contourlet-toolbox.

The methods
- MakeONFilter
- MirrorFilt

were taken from WaveLab850 (http://www-stat.stanford.edu/~wavelab/).


For more details, see: "ShearLab 3D: Faithful Digital Shearlet Transforms based on Compactly Supported Shearlets"

[1] A. L. da Cunha, J. Zhou, M. N. Do, "The Nonsubsampled Contourlet Transform: Theory, Design, and Applications," IEEE Transactions on Image Processing, 2005.