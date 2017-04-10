"""
This module contains all utilitiy files provided by the ShearLab3D toolbox
from MATLAB such as padding arrays, the discretized shear operator et cetera.

All these functions were originally written by Rafael Reisenhofer and are
published in the ShearLab3Dv11 toolbox on http://www.shearlet.org.

Stefan Loock, February 2, 2017 [sloock@gwdg.de]
"""

import sys
import numpy as np
import scipy as scipy
import scipy.io as sio
from pyshearlab.pySLFilters import *



def SLcheckFilterSizes(rows,cols, shearLevels,directionalFilter,scalingFilter,
                        waveletFilter,scalingFilter2):
    """
    Checks filter sizes for different configurations for a given size of a
    square image with rows and cols given by the first two arguments. The
    argument shearLevels is a vector containing the desired shear levels for
    the shearlet transform.
    """
    directionalFilter = directionalFilter
    scalingFilter = scalingFilter
    waveletFilter = waveletFilter
    scalingFilter2 = scalingFilter2

    filterSetup = [None] * 8

    # configuration 1
    filterSetup[0] = {"directionalFilter": directionalFilter,
                        "scalingFilter": scalingFilter,
                        "waveletFilter": waveletFilter,
                        "scalingFilter2": scalingFilter2}
    # configuration 2
    h0, h1 = dfilters('dmaxflat4', 'd')/np.sqrt(2)
    directionalFilter = modulate2(h0, 'c')
    scalingFilter = np.array([0.0104933261758410,-0.0263483047033631,
                    -0.0517766952966370, 0.276348304703363, 0.582566738241592,
                    0.276348304703363,-0.0517766952966369,-0.0263483047033631,
                    0.0104933261758408])
    waveletFilter = MirrorFilt(scalingFilter)
    scalingFilter2 = scalingFilter
    filterSetup[1] = {"directionalFilter": directionalFilter,
                        "scalingFilter": scalingFilter,
                        "waveletFilter": waveletFilter,
                        "scalingFilter2": scalingFilter2}
    # configuration 3
    h0, h1 = dfilters('cd', 'd')/np.sqrt(2)
    directionalFilter = modulate2(h0, 'c')
    scalingFilter = np.array([0.0104933261758410, -0.0263483047033631,
                    -0.0517766952966370, 0.276348304703363, 0.582566738241592,
                    0.276348304703363, -0.0517766952966369,-0.0263483047033631,
                    0.0104933261758408])
    waveletFilter = MirrorFilt(scalingFilter)
    scalingFilter2 = scalingFilter
    filterSetup[2] = {"directionalFilter": directionalFilter,
                        "scalingFilter": scalingFilter,
                        "waveletFilter": waveletFilter,
                        "scalingFilter2": scalingFilter2}
    # configuration 4 - somehow the same as 3, i don't know why?!
    h0, h1 = dfilters('cd', 'd')/np.sqrt(2)
    directionalFilter = modulate2(h0, 'c')
    scalingFilter = np.array([0.0104933261758410, -0.0263483047033631,
                    -0.0517766952966370, 0.276348304703363, 0.582566738241592,
                    0.276348304703363, -0.0517766952966369,-0.0263483047033631,
                    0.0104933261758408])
    waveletFilter = MirrorFilt(scalingFilter)
    scalingFilter2 = scalingFilter
    filterSetup[3] = {"directionalFilter": directionalFilter,
                        "scalingFilter": scalingFilter,
                        "waveletFilter": waveletFilter,
                        "scalingFilter2": scalingFilter2}
    # configuration 5
    h0, h1 = dfilters('cd', 'd')/np.sqrt(2)
    directionalFilter = modulate2(h0, 'c')
    scalingFilter = MakeONFilter('Coiflet', 1)
    waveletFilter = MirrorFilt(scalingFilter)
    scalingFilter2 = scalingFilter
    filterSetup[4] = {"directionalFilter": directionalFilter,
                        "scalingFilter": scalingFilter,
                        "waveletFilter": waveletFilter,
                        "scalingFilter2": scalingFilter2}
    # configuration 6
    h0, h1 = dfilters('cd', 'd')/np.sqrt(2)
    directionalFilter = modulate2(h0, 'c')
    scalingFilter = MakeONFilter('Daubechies', 4)
    waveletFilter = MirrorFilt(scalingFilter)
    scalingFilter2 = scalingFilter
    filterSetup[5] = {"directionalFilter": directionalFilter,
                        "scalingFilter": scalingFilter,
                        "waveletFilter": waveletFilter,
                        "scalingFilter2": scalingFilter2}
    # configuration 7
    h0, h1 = dfilters('oqf_362', 'd')/np.sqrt(2)
    directionalFilter = modulate2(h0, 'c')
    scalingFilter = MakeONFilter('Daubechies', 4)
    waveletFilter = MirrorFilt(scalingFilter)
    scalingFilter2 = scalingFilter
    filterSetup[6] = {"directionalFilter": directionalFilter,
                        "scalingFilter": scalingFilter,
                        "waveletFilter": waveletFilter,
                        "scalingFilter2": scalingFilter2}
    # configuration 8
    h0, h1 = dfilters('oqf_362', 'd')/np.sqrt(2)
    directionalFilter = modulate2(h0, 'c')
    scalingFilter = MakeONFilter('Haar')
    scalingFilter2 = scalingFilter
    filterSetup[7] = {"directionalFilter": directionalFilter,
                        "scalingFilter": scalingFilter,
                        "waveletFilter": waveletFilter,
                        "scalingFilter2": scalingFilter2}
    success = False
    for k in range(8):
        #check 1
        lwfilter = filterSetup[k]["waveletFilter"].size
        lsfilter = filterSetup[k]["scalingFilter"].size
        lcheck1 = lwfilter
        for j in range(shearLevels.size):
            lcheck1 = lsfilter + 2*lcheck1 -2
        if lcheck1 > cols or lcheck1 > rows:
            continue
        #check 2
        rowsdirfilter = np.asarray(filterSetup[k]["directionalFilter"].shape)[0]
        colsdirfilter = np.asarray(filterSetup[k]["directionalFilter"].shape)[1]
        lcheck2 = (rowsdirfilter-1)*np.power(2, max(shearLevels)+1)+1

        lsfilter2 = filterSetup[k]["scalingFilter2"].size
        lcheck2help = lsfilter2
        for j in range(1, int(max(shearLevels))+1):
            lcheck2help = lsfilter2 + 2*lcheck2help -2
        lcheck2 = lcheck2help + lcheck2-1
        if lcheck2 > cols or lcheck2 > rows or colsdirfilter > cols or colsdirfilter > rows:
            continue
        success = 1
        break
    directionalFilter = filterSetup[k]["directionalFilter"]
    scalingFilter = filterSetup[k]["scalingFilter"]
    waveletFilter = filterSetup[k]["waveletFilter"]
    scalingFilter2 = filterSetup[k]["scalingFilter2"]
    if success == 0:
        sys.exit("The specified Shearlet system is not available for data of size "
            + str(rows) + "x" + str(cols) + ". Try decreasing the number of scales and shearings.")
    if success == 1 and k>1:
        print("Warning: The specified Shearlet system was not available for data of size " + str(rows) + "x" + str(cols) + ". Filters were automatically set to configuration " + str(k) + "(see SLcheckFilterSizes).")
        return directionalFilter, scalingFilter, waveletFilter, scalingFilter2
    else:
        return directionalFilter, scalingFilter, waveletFilter, scalingFilter2



def SLcomputePSNR(X, Xnoisy):
    """
    SLcomputePSNR Compute peak signal to noise ratio (PSNR).

    Usage:

        PSNR = SLcomputePSNR(X, Xnoisy)

    Input:

        X:      2D or 3D signal.
        Xnoisy: 2D or 3D noisy signal.

    Output:

        PSNR: The peak signal to noise ratio (in dB).
    """

    MSEsqrt = np.linalg.norm(X-Xnoisy) / np.sqrt(X.size)
    if MSEsqrt == 0:
        return np.inf
    else:
        return 20 * np.log10(255 / MSEsqrt)

def SLcomputeSNR(X, Xnoisy):
    """
    SLcomputeSNR Compute signal to noise ratio (SNR).

    Usage:

        SNR = SLcomputeSNR(X, Xnoisy)

    Input:

        X:      2D or 3D signal.
        Xnoisy: 2D or 3D noisy signal.

    Output:

        SNR: The signal to noise ratio (in dB).
    """

    if np.linalg.norm(X-Xnoisy) == 0:
        return np.Inf
    else:
        return 10 * np.log10( np.sum(np.power(X,2)) / np.sum(np.power(X-Xnoisy,2)) )



def SLdshear(inputArray, k, axis):
    """
    Computes the discretized shearing operator for a given inputArray, shear
    number k and axis.

    This version is adapted such that the MATLAB indexing can be used here in the
    Python version.
    """
    axis = axis - 1
    if k==0:
        return inputArray
    rows = np.asarray(inputArray.shape)[0]
    cols = np.asarray(inputArray.shape)[1]

    shearedArray = np.zeros((rows, cols), dtype=inputArray.dtype)

    if axis == 0:
        for col in range(cols):
            shearedArray[:,col] = np.roll(inputArray[:,col], int(k * np.floor(cols/2-col)))
    else:
        for row in range(rows):
            shearedArray[row,:] = np.roll(inputArray[row,:], int(k * np.floor(rows/2-row)))
    return shearedArray


def SLgetShearletIdxs2D(shearLevels, full=0, *args):
    """
    Computes a index set describing a 2D shearlet system.

    Usage:

        shearletIdxs = SLgetShearletIdxs2D(shearLevels)
        shearletIdxs = SLgetShearletIdxs2D(shearLevels, full)
        shearletIdxs = SLgetShearletIdxs2D(shearLevels, full, 'NameRestriction1', ValueRestriction1,...)

    Input:

        shearLevels: A 1D array, specifying the level of shearing on
                     each scale.
                     Each entry of shearLevels has to be >= 0. A
                     shear level of K means that the generating
                     shearlet is sheared 2^K times in each direction
                     for each cone.

                     For example: If shearLevels = [1 1 2], the
                     corresponding shearlet system has a maximum
                     redundancy of
                     (2*(2*2^1+1))+(2*(2*2^1+1))+(2*(2*2^2+1))=38
                     (omitting the lowpass shearlet). Note
                     that it is recommended not to use the full
                     shearlet system but to omit shearlets lying on
                     the border of the second cone as they are only
                     slightly different from those on the border of
                     the first cone.

               full: Logical value that determines whether the
                     indexes are computed for a full shearlet
                     system or if shearlets lying on the border of
                     the second cone are omitted. The default and
                     recommended value is 0.

        TypeRestriction1: Possible restrictions: 'cones', 'scales',
                     'shearings'.

        ValueRestriction1: Numerical value or Array specifying a
                     restriction. If the type of the restriction is
                     'scales' the value 1:2 ensures that only indexes
                     corresponding the shearlets on the first two
                     scales are computed.

    Output:

        shearletIdxs: Nx3 matrix, where each row describes one
                    shearlet in the format [cone scale shearing].

    Example 1:
        Compute the indexes, describing a 2D shearlet system with 3 scales:

        shearletIdxs = SLgetShearletIdxs2D([1 1 2])

    Example 2:
        Compute the subset of a shearlet system, containing only shearlets on
        the first scale:

        shearletSystem = SLgetShearletSystem2D(0,512,512,4)
        subsetIdxs = SLgetShearletIdxs2D(shearletSystem.shearLevels,shearletSystem.full,'scales',1)
        subsystem = SLgetSubsystem2D(shearletSystem,subsetIdxs)


    See also: SLgetShearletSystem2D, SLgetSubsystem2D
    """
    shearletIdxs = []
    includeLowpass = 1
    # if a scalar is passed as shearLevels, we treat it as an array.
    if not hasattr(shearLevels, "__len__"):
        shearLevels = np.array([shearLevels])
    scales = np.asarray(range(1,len(shearLevels)+1))
    shearings = np.asarray(range(-np.power(2,np.max(shearLevels)),np.power(2,np.max(shearLevels))+1))
    cones = np.array([1,2])
    for j in range(0,len(args),2):
        includeLowpass = 0
        if args[j] == "scales":
            scales = args[j+1]
        elif args[j] == "shearings":
            shearings = args[j+1]
        elif args[j] == "cones":
            cones = args[j+1]
    for cone in np.intersect1d(np.array([1,2]), cones):
        for scale in np.intersect1d(np.asarray(range(1,len(shearLevels)+1)), scales):
            for shearing in np.intersect1d(np.asarray(range(-np.power(2,shearLevels[scale-1]),np.power(2,shearLevels[scale-1])+1)), shearings):
                if (full == 1) or (cone == 1) or (np.abs(shearing) < np.power(2, shearLevels[scale-1])):
                    shearletIdxs.append(np.array([cone, scale, shearing]))
    if includeLowpass or 0 in scales or 0 in cones:
        shearletIdxs.append(np.array([0,0,0]))
    return np.asarray(shearletIdxs)



def SLgetShearlets2D(preparedFilters, shearletIdxs=None):
    """
    Compute 2D shearlets in the frequency domain.

    Usage:

        [shearlets, RMS, dualFrameWeights]
            = SLgetShearlets2D(preparedFilters)
        [shearlets, RMS, dualFrameWeights]
            = SLgetShearlets2D(preparedFilters, shearletIdxs)

    Input:

        preparedFilters: A structure containing filters that can be
                         used to compute 2D shearlets. Such filters
                         can be generated with SLprepareFilters2D.

        shearletdIdxs: A Nx3 array, specifying each shearlet that
                        is to be computed in the format
                        [cone scale shearing] where N denotes the
                        number of shearlets. The vertical cone in
                        time domain is indexed by 1 while the
                        horizontal cone is indexed by 2.
                        Note that the values for scale and shearing
                        are limited by the precomputed filters. The
                        lowpass shearlet is indexed by [0 0 0]. If
                        no shearlet indexes are specified,
                        SLgetShearlets2D returns a standard
                        shearlet system based on the precomputed
                        filters.
                        Such a standard index set can also be
                        obtained by calling SLgetShearletIdxs2D.

    Output:

         shearlets: A X x Y x N array of N 2D shearlets in the
                    frequency domain where X and Y denote the
                    size of each shearlet.
               RMS: A 1xN array containing the root mean
                    squares (L2-norm divided by sqrt(X*Y)) of all
                    shearlets stored in shearlets. These values
                    can be used to normalize shearlet coefficients
                    to make them comparable.
        dualFrameWeights: A X x Y matrix containing the absolute and
                    squared sum over all shearlets stored in
                    shearlets. These weights are needed to compute
                    the dual frame during reconstruction.

    Description:

    The wedge and bandpass filters in preparedFilters are used to compute
    shearlets on different scales and of different shearings, as specified by
    the shearletIdxs array. Shearlets are computed in the frequency domain.
    To get the i-th shearlet in the time domain, use

            fftshift(ifft2(ifftshift(shearlets(:,:,i)))).

    Each Shearlet is centered at floor([X Y]/2) + 1.

    Example 1:
    Compute the lowpass shearlet:

        preparedFilters
            = SLprepareFilters2D(512,512,4,[1 1 2 2])
        lowpassShearlet
            = SLgetShearlets2D(preparedFilters,[0 0 0])
        lowpassShearletTimeDomain
            = fftshift(ifft2(ifftshift(lowpassShearlet)))

    Example 2:
    Compute a standard shearlet system of four scales:

        preparedFilters = SLprepareFilters2D(512,512,4)
        shearlets = SLgetShearlets2D(preparedFilters)

    Example 3:
    Compute a full shearlet system of four scales:

        preparedFilters = SLprepareFilters2D(512,512,4)
        shearlets = SLgetShearlets2D(preparedFilters,SLgetShearletIdxs2D(preparedFilters.shearLevels,1))

    See also: SLprepareFilters2D, SLgetShearletIdxs2D, SLsheardec2D, SLshearrec2D
    """

    if shearletIdxs is None:
        shearletIdxs = SLgetShearletIdxs2D(preparedFilters["shearLevels"])
    # useGPU = preparedFilters["useGPU"] - we don't support gpus right now
    rows = preparedFilters["size"][0]
    cols = preparedFilters["size"][1]
    nShearlets = shearletIdxs.shape[0]
    # allocate shearlets
    # ...skipping gpu part...
    shearlets = np.zeros((rows,cols,nShearlets), dtype=complex)

    # compute shearlets
    for j in range(nShearlets):
        cone = shearletIdxs[j,0]
        scale = shearletIdxs[j,1]
        shearing = shearletIdxs[j,2]
        if cone == 0:
            shearlets[:,:,j] = preparedFilters["cone1"]["lowpass"]
        elif cone == 1:
            #here, the fft of the digital shearlet filters described in
            #equation (23) on page 15 of "ShearLab 3D: Faithful Digital
            #Shearlet Transforms based on Compactly Supported Shearlets" is computed.
            #for more details on the construction of the wedge and bandpass
            #filters, please refer to SLgetWedgeBandpassAndLowpassFilters2D.
            #print(preparedFilters["cone1"]["wedge"][preparedFilters["shearLevels"][scale-1]])
            #print(preparedFilters["shearLevels"][scale-1])
            # letztes index checken! ggf. +1?
            shearlets[:,:,j] = preparedFilters["cone1"]["wedge"][preparedFilters["shearLevels"][scale-1]][:,:,-shearing+np.power(2,preparedFilters["shearLevels"][scale-1])]*np.conj(preparedFilters["cone1"]["bandpass"][:,:,scale-1])
        else:
            shearlets[:,:,j] = np.transpose(preparedFilters["cone2"]["wedge"][preparedFilters["shearLevels"][scale-1]][:,:,shearing+np.power(2,preparedFilters["shearLevels"][scale-1])]*np.conj(preparedFilters["cone2"]["bandpass"][:,:,scale-1]))
        # the matlab version only returns RMS and dualFrameWeights if the function
        # is called accordingly. we compute them always for the time being.
        RMS = np.linalg.norm(shearlets, axis=(0,1))/np.sqrt(rows*cols)
        dualFrameWeights = np.sum(np.power(np.abs(shearlets),2), axis=2)

    return shearlets, RMS, dualFrameWeights


def SLgetWedgeBandpassAndLowpassFilters2D(rows,cols,shearLevels,directionalFilter=None,scalingFilter=None,waveletFilter=None,scalingFilter2=None):
    """
    Computes the wedge, bandpass and lowpass filter for 2D shearlets. If no
    directional filter, scaling filter and wavelet filter are given, some
    standard filters are used.

    rows, cols and shearLevels are mandatory.
    """
    if scalingFilter is None:
        scalingFilter = np.array([0.0104933261758410, -0.0263483047033631,
                            -0.0517766952966370, 0.276348304703363,
                            0.582566738241592, 0.276348304703363,
                            -0.0517766952966369, -0.0263483047033631,
                            0.0104933261758408])
    if scalingFilter2 is None:
        scalingFilter2 = scalingFilter
    if waveletFilter is None:
        waveletFilter = MirrorFilt(scalingFilter)
    if directionalFilter is None:
        h0,h1 = dfilters('dmaxflat4', 'd')/np.sqrt(2)
        directionalFilter = modulate2(h0, 'c')

###########################################################################
#   all page and equation numbers refer to "ShearLab 3D: Faithful Digital #
#   Shearlet Transforms based on Compactly Supported Shearlets"           #
###########################################################################

    # initialize variables

    # get number of scales
    NScales = shearLevels.size

    # allocate bandpass and wedge filters
    bandpass = np.zeros((rows,cols, NScales), dtype=complex) #these filters partition the frequency plane into different scales
    wedge = [None] * ( max(shearLevels) + 1 ) # these filters partition the frequenecy plane into different directions

    #normalize filters
    directionalFilter = directionalFilter/sum(sum(np.absolute(directionalFilter)))

    ## compute 1D high and lowpass filters at different scales:
    #
    # filterHigh{NScales} = g_1 and filterHigh{1} = g_J (compare page 11)
    filterHigh = [None] * NScales
    # we have filterLow{NScales} = h_1 and filterLow{1} = h_J (compare page 11)
    filterLow = [None] * NScales
    #typically, we have filterLow2{max(shearLevels)+1} = filterLow{NScales},
    # i.e. filterLow2{NScales} = h_1 (compare page 11)
    filterLow2 = [None] * (max(shearLevels) + 1)

    ## initialize wavelet highpass and lowpass filters:
    #
    # this filter is typically chosen to form a quadrature mirror filter pair
    # with scalingFilter and corresponds to g_1 on page 11
    filterHigh[-1] = waveletFilter
    filterLow[-1] = scalingFilter # this filter corresponds to h_1 on page 11
    # this filter is typically chosen to be equal to scalingFilter and provides
    # the y-direction for the tensor product constructing the 2D wavelet filter
    # w_j on page 14
    filterLow2[-1] = scalingFilter2

    # compute wavelet high- and lowpass filters associated with a 1D Digital
    # wavelet transform on Nscales scales, e.g., we compute h_1 to h_J and
    # g_1 to g_J (compare page 11) with J = nScales.
    for j in range(len(filterHigh)-2,-1,-1):
        filterLow[j] = np.convolve(filterLow[-1], SLupsample(filterLow[j+1],2,1))
        filterHigh[j] = np.convolve(filterLow[-1], SLupsample(filterHigh[j+1],2,1))
    for j in range(len(filterLow2)-2,-1,-1):
        filterLow2[j] = np.convolve(filterLow2[-1], SLupsample(filterLow2[j+1],2,1))
    # construct bandpass filters for scales 1 to nScales
    for j in range(len(filterHigh)):
        bandpass[:,:,j] = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(SLpadArray(filterHigh[j], np.array([rows, cols])))))

    ## construct wedge filters for achieving directional selectivity.
    # as the entries in the shearLevels array describe the number of differently
    # sheared atoms on a certain scale, a different set of wedge
    # filters has to be constructed for each value in shearLevels.
    filterLow2[-1].shape = (1, len(filterLow2[-1]))
    for shearLevel in np.unique(shearLevels):
            # preallocate a total of floor(2^(shearLevel+1)+1) wedge filters, where
            # floor(2^(shearLevel+1)+1) is the number of different directions of
            # shearlet atoms associated with the horizontal (resp. vertical)
            # frequency cones.
            #
            # plus one for one unsheared shearlet
            wedge[shearLevel] = np.zeros((rows, cols, int(np.floor(np.power(2,shearLevel+1)+1))), dtype=complex)

            # upsample directional filter in y-direction. by upsampling the directional
            # filter in the time domain, we construct repeating wedges in the
            # frequency domain ( compare abs(fftshift(fft2(ifftshift(directionalFilterUpsampled)))) and
            # abs(fftshift(fft2(ifftshift(directionalFilter)))) ).

            directionalFilterUpsampled = SLupsample(directionalFilter, 1, np.power(2,shearLevel+1)-1)

            # remove high frequencies along the y-direction in the frequency domain.
            # by convolving the upsampled directional filter with a lowpass filter in y-direction, we remove all
            # but the central wedge in the frequency domain.
            #
            # convert filterLow2 into a pseudo 2D array of size (len, 1) to use
            # the scipy.signal.convolve2d accordingly.
            filterLow2[-1-shearLevel].shape = (1, len(filterLow2[-1-shearLevel]))

            wedgeHelp = scipy.signal.convolve2d(directionalFilterUpsampled,np.transpose(filterLow2[len(filterLow2)-shearLevel-1]));
            wedgeHelp = SLpadArray(wedgeHelp,np.array([rows,cols]));
            # please note that wedgeHelp now corresponds to
            # conv(p_j,h_(J-j*alpha_j/2)') in the language of the paper. to see
            # this, consider the definition of p_j on page 14, the definition of w_j
            # on the same page an the definition of the digital sheralet filter on
            # page 15. furthermore, the g_j part of the 2D wavelet filter w_j is
            # invariant to shearings, hence it suffices to apply the digital shear
            # operator to wedgeHelp.

            ## application of the digital shear operator (compare equation (22))
            # upsample wedge filter in x-direction. this operation corresponds to
            # the upsampling in equation (21) on page 15.
            wedgeUpsampled = SLupsample(wedgeHelp,2,np.power(2,shearLevel)-1);

            #convolve wedge filter with lowpass filter, again following equation
            # (21) on page 14.
            #print("shearLevel:" + str(shearLevel) + ", Index: " + str(len(filterLow2)-max(shearLevel-1,0)-1) + ", Shape: " + str(filterLow2[len(filterLow2)-max(shearLevel-1,0)-1].shape))
            #print(filterLow2[len(filterLow2)-max(shearLevel-1,0)-1].shape)
            lowpassHelp = SLpadArray(filterLow2[len(filterLow2)-max(shearLevel-1,0)-1], np.asarray(wedgeUpsampled.shape))
            if shearLevel >= 1:
                wedgeUpsampled = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(lowpassHelp))) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wedgeUpsampled))))))
            lowpassHelpFlip = np.fliplr(lowpassHelp)
            # traverse all directions of the upper part of the left horizontal
            # frequency cone
            for k in range(-np.power(2, shearLevel), np.power(2, shearLevel)+1):
                # resample wedgeUpsampled as given in equation (22) on page 15.
                wedgeUpsampledSheared = SLdshear(wedgeUpsampled,k,2)
                # convolve again with flipped lowpass filter, as required by
                # equation (22) on page 15
                if shearLevel >= 1:
                        wedgeUpsampledSheared = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(lowpassHelpFlip))) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wedgeUpsampledSheared))))))
                # obtain downsampled and renormalized and sheared wedge filter
                # in the frequency domain, according to equation (22), page 15.
                wedge[shearLevel][:,:,int(np.fix(np.power(2,shearLevel))-k)] = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.power(2,shearLevel)*wedgeUpsampledSheared[:,0:np.power(2,shearLevel)*cols-1:np.power(2,shearLevel)])))
    # compute low pass filter of shearlet system
    lowpass = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(SLpadArray(np.outer(filterLow[0],filterLow[0]), np.array([rows, cols])))))
    return wedge, bandpass, lowpass


def SLnormalizeCoefficients2D(coeffs, shearletSystem):
    """
    Normalizes the shearlet coefficients coeffs for a given set of 2D
    shearlet coefficients and a given shearlet system shearletSystem
    by dividing by the RMS of each shearlet.
    """
    coeffsNormalized = np.zeros(coeffs.shape)

    for i in range(shearletSystem["nShearlets"]):
        coeffsNormalized[:,:,i] = coeffs[:,:,i] / shearletSystem["RMS"][i]
    return coeffsNormalized




def SLpadArray(array, newSize):
    """
    Implements the padding of an array as performed by the Matlab variant.
    """
    if np.isscalar(newSize):
        #padSizes = np.zeros((1,newSize))
        # check if array is a vector...
        currSize = array.size
        paddedArray = np.zeros(newSize)
        sizeDiff = newSize - currSize
        idxModifier = 0
        if sizeDiff < 0:
            sys.exit("Error: newSize is smaller than actual array size.")
        if sizeDiff == 0:
            print("Warning: newSize is equal to padding size.")
        if sizeDiff % 2 == 0:
            padSizes = sizeDiff//2
        else:
            padSizes = int(np.ceil(sizeDiff/2))
            if currSize % 2 == 0:
                # index 1...k+1
                idxModifier = 1
            else:
                # index 0...k
                idxModifier = 0
        print(padSizes)
        paddedArray[padSizes-idxModifier:padSizes+currSize-idxModifier] = array

    else:
        padSizes = np.zeros(newSize.size)
        paddedArray = np.zeros((newSize[0], newSize[1]))
        idxModifier = np.array([0, 0])
        currSize = np.asarray(array.shape)
        if array.ndim == 1:
            currSize = np.array([len(array), 0])
        for k in range(newSize.size):
            sizeDiff = newSize[k] - currSize[k]
            if sizeDiff < 0:
                sys.exit("Error: newSize is smaller than actual array size in dimension " + str(k) + ".")
            if sizeDiff == 0:
                print("Warning: newSize is equal to padding size in dimension " + str(k) + ".")
            if sizeDiff % 2 == 0:
                padSizes[k] = sizeDiff//2
            else:
                padSizes[k] = np.ceil(sizeDiff/2)
                if currSize[k] % 2 == 0:
                    # index 1...k+1
                    idxModifier[k] = 1
                else:
                    # index 0...k
                    idxModifier[k] = 0
        padSizes = padSizes.astype(int)

        # if array is 1D but paddedArray is 2D we simply put the array (as a
        # row array in the middle of the new empty array). this seems to be
        # the behavior of the ShearLab routine from matlab.
        if array.ndim == 1:
            paddedArray[padSizes[1], padSizes[0]:padSizes[0]+currSize[0]+idxModifier[0]] = array
        else:
            paddedArray[padSizes[0]-idxModifier[0]:padSizes[0]+currSize[0]-idxModifier[0],
                    padSizes[1]:padSizes[1]+currSize[1]+idxModifier[1]] = array
    return paddedArray


def SLprepareFilters2D(rows, cols, nScales, shearLevels=None,
                directionalFilter=None, scalingFilter=None, waveletFilter=None,
                scalingFilter2=None):
    """
    Usage:

        filters = SLprepareFilters2D(rows, cols, nScales)
        filters = SLprepareFilters2D(rows, cols, nScales,
                                            shearLevels)
        filters = SLprepareFilters2D(rows, cols, nScales,
                            shearLevels, directionalFilter)
        filters
            = SLprepareFilters2D(rows, cols, nScales, shearLevels,
                directionalFilter, quadratureMirrorfilter)

 Input:

        rows: Number of rows.
        cols: Number of columns.
        nScales: Number of scales of the desired shearlet system.
                Has to be >= 1.
        shearLevels: A 1xnScales sized array, specifying the level
                    of shearing occuring on each scale. Each entry
                    of shearLevels has to be >= 0. A shear level
                    of K means that the generating shearlet is
                    sheared 2^K times in each direction for each
                    cone.
                    For example: If nScales = 3 and
                    shearLevels = [1 1 2], the precomputed filters
                    correspond to a shearlet system with a maximum
                    number of
                    (2*(2*2^1+1))+(2*(2*2^1+1))+(2*(2*2^2+1))=38
                    shearlets (omitting the lowpass shearlet and
                    translation). Note that it is recommended not
                    to use the full shearlet system but to omit
                    shearlets lying on the border of the second
                    cone as they are only slightly different from
                    those on the border of the first cone.
        directionalFilter: A 2D directional filter that serves as
                    the basis of the directional 'component' of
                    the shearlets.
                    The default choice is
                        modulate2(dfilters('dmaxflat4','d'),'c').
                    For small sized inputs, or very large systems
                    the default directional filter might be too
                    large. In this case, it is recommended to use
                        modulate2(dfilters('cd','d'),'c').
        quadratureMirrorFilter: A 1D quadrature mirror filter
                    defining the wavelet 'component' of the
                    shearlets. The default choice is
                    [0.0104933261758410,-0.0263483047033631,-0.0517766952966370,
                 0.276348304703363,0.582566738241592,0.276348304703363,
                 -0.0517766952966369,-0.0263483047033631,0.0104933261758408].

                    Other QMF filters can be genereted with
                    MakeONFilter.

 Output:

        filters: A structure containing wedge and bandpass filters
                    that can be used to compute 2D shearlets.

 Description:

 Based on the specified directional filter and quadrature mirror filter,
 2D wedge and bandpass filters are computed that can be used to compute arbitrary 2D
 shearlets for data of size [rows cols] on nScales scales with as many
 shearings as specified by the shearLevels array.

 Example 1:

 Prepare filters for a input of size 512x512 and a 4-scale shearlet system

        preparedFilters = SLprepareFilters2D(512,512,4)
        shearlets = SLgetShearlets2D(preparedFilters)

 Example 2:

 Prepare filters for a input of size 512x512 and a 3-scale shearlet system
 with 2^3 = 8 shearings in each direction for each cone on all 3 scales.

        preparedFilters = SLprepareFilters2D(512,512,3,[3 3 3])
        shearlets = SLgetShearlets2D(preparedFilters)

 See also: SLgetShearletIdxs2D,SLgetShearlets2D,dfilters,MakeONFilter
    """
# check input arguments
    if shearLevels is None:
        shearLevels = np.ceil(np.arange(1,nScales+1)/2).astype(int)
    if scalingFilter is None:
        scalingFilter = np.array([0.0104933261758410, -0.0263483047033631,
                        -0.0517766952966370, 0.276348304703363, 0.582566738241592,
                        0.276348304703363, -0.0517766952966369, -0.0263483047033631,
                        0.0104933261758408])
    if scalingFilter2 is None:
        scalingFilter2 = scalingFilter
    if directionalFilter is None:
        h0, h1 = dfilters('dmaxflat4', 'd')/np.sqrt(2)
        directionalFilter = modulate2(h0, 'c')
    if waveletFilter is None:
        waveletFilter = MirrorFilt(scalingFilter)
    directionalFilter, scalingFilter, waveletFilter, scalingFilter2 = SLcheckFilterSizes(rows, cols, shearLevels, directionalFilter, scalingFilter, waveletFilter, scalingFilter2)
    fSize = np.array([rows, cols])
    filters = {"size": fSize, "shearLevels": shearLevels}
    wedge1, bandpass1, lowpass1 = SLgetWedgeBandpassAndLowpassFilters2D(rows,cols,shearLevels,directionalFilter,scalingFilter,waveletFilter,scalingFilter2)
    wedge1[0] = 0   # for matlab compatibilty (saving filters as .mat files)
    filters["cone1"] = {"wedge": wedge1, "bandpass": bandpass1, "lowpass": lowpass1}
    if rows == cols:
        filters["cone2"] = filters["cone1"]
    else:
        wedge2, bandpass2, lowpass2 = SLgetWedgeBandpassAndLowpassFilters2D(cols,rows,shearLevels,directionalFilter,scalingFilter,waveletFilter,scalingFilter2)
        wedge2[0] = 0   # for matlab compatibilty (saving filters as .mat files)
        filters["cone2"] = {"wedge": wedge2, "bandpass": bandpass2, "lowpass": lowpass2}
    return filters
#
##############################################################################


##############################################################################
#
def SLupsample(array, dims, nZeros):
    """
    Performs an upsampling by a number of nZeros along the dimenion(s) dims
    for a given array.

    Note that this version behaves like the Matlab version, this means we would
    have dims = 1 or dims = 2 instead of dims = 0 and dims = 1.
    """
    if array.ndim == 1:
        sz = len(array)
        idx = range(1,sz)
        arrayUpsampled = np.insert(array, idx, 0)
    else:
        sz = np.asarray(array.shape)
        # behaves like in matlab: dims == 1 and dims == 2 instead of 0 and 1.
        if dims == 0:
            sys.exit("SLupsample behaves like in Matlab, so chose dims = 1 or dims = 2.")
        if dims == 1:
            arrayUpsampled = np.zeros(((sz[0]-1)*(nZeros+1)+1, sz[1]))
            for col in range(sz[0]):
                arrayUpsampled[col*(nZeros)+col,:] = array[col,:]
        if dims == 2:
            arrayUpsampled = np.zeros((sz[0], ((sz[1]-1)*(nZeros+1)+1)))
            for row in range(sz[1]):
                arrayUpsampled[:,row*(nZeros)+row] = array[:,row]
    return arrayUpsampled

#
##############################################################################
