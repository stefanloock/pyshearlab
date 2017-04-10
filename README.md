# pyShearLab
pyShearLab is a Python toolbox which is based on [ShearLab3D](http://www3.math.tu-berlin.de/numerik/www.shearlab.org/software) written by [Rafael Reisenhofer](http://www.math.uni-bremen.de/~reisenho/) and has been ported to Python by Stefan Loock.

Currently, pyShearLab only offers a two-dimensional subset of ShearLab3D which contains both 2D and 3D transforms.

## Dependencies
The toolbox needs the following Python packages in order to work properly:

* [NumPy](http://www.numpy.org/)
* [SciPy](https://scipy.org/)
* [Matplotlib](http://matplotlib.org/)
* [Pillow (PIL)](https://python-pillow.org/)

pyShearLab2D has been developed and tested with Python 3.6 using the Anaconda package on Linux (Ubuntu 16.04.2 LTS), Windows 10 and Mac OS X (10.11-10.12). There are issues when using pyShearLab2d with Python 2.X.

## Installation
You can simply download, unzip and use pyShearLab. Depending on your specific Python development environment, you may want to add the pyShearLab2D folder to your Python environment (Python Path). The dependencies can be installed using pip. If you use Anaconda, they are already installed.
A pip package is currently _not_ available, sorry.

## Usage
In order to use pyShearLab you need to import it as a module, see pySLExampleDenoising.py as an example. The denoising example 
provides all neccessary steps to understand how to use the toolbox. When using the transform in an iterative scheme, the 
creation of shearlet system can be done in a pre-processing step which significantly speeds up the process.

Please note that the images have to be square in size.

## Copyright
pyShearLab was written by Stefan Loock who acknowledges funding by the [SFB 755 Nanoscale Photonic Imaging](http://www.uni-goettingen.de/de/318955.html). pyShearLab is based
on [ShearLab3D](http://www3.math.tu-berlin.de/numerik/www.shearlab.org/software)  which is written by [Rafael Reisenhofer](http://www.math.uni-bremen.de/~reisenho/)  and published under the GPL. The toolbox uses some functions from:

* [WaveLab 850](http://statweb.stanford.edu/~wavelab/): Copyright (c) 1993-5. Jonathan Buckheit and David Donoho
* [Nonsubsampled Contourlet Toolbox](https://www.mathworks.com/matlabcentral/fileexchange/10049-nonsubsampled-contourlet-toolbox): Copyright (c) 2004. Arthur L. da Cunha

which have been translated to Python.
