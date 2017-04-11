"""
This module contains all neccessary files to compute the filters used
in the pyShearLab2D toolbox. Most of these files are taken from different
MATLAB toolboxes and were translated to Python. Credit is given in each
individual function.


Stefan Loock, February 2, 2017 [sloock@gwdg.de]
"""

import numpy as np
from scipy import signal as signal


try:
    import pyfftw
    fftlib = pyfftw.interfaces.numpy_fft
    pyfftw.interfaces.cache.enable()
except ImportError:
    fftlib = np.fft
    

def MakeONFilter(Type,Par=1):
    """
    This is a rewrite of the original Matlab implementation of MakeONFilter.m
    from the WaveLab850 toolbox.

    MakeONFilter -- Generate Orthonormal QMF Filter for Wavelet Transform

    Usage:

        qmf = MakeONFilter(Type, Par)

    Inputs:

        Type:  string: 'Haar', 'Beylkin', 'Coiflet', 'Daubechies',
                        'Symmlet', 'Vaidyanathan', 'Battle'

    Outputs:

        qmf:    quadrature mirror filter

    Description

    The Haar filter (which could be considered a Daubechies-2) was the
    first wavelet, though not called as such, and is discontinuous.

    The Beylkin filter places roots for the frequency response function
    close to the Nyquist frequency on the real axis.

    The Coiflet filters are designed to give both the mother and father
    wavelets 2*Par vanishing moments; here Par may be one of 1,2,3,4 or 5.

    The Daubechies filters are minimal phase filters that generate wavelets
    which have a minimal support for a given number of vanishing moments.
    They are indexed by their length, Par, which may be one of
    4,6,8,10,12,14,16,18 or 20. The number of vanishing moments is par/2.

    Symmlets are also wavelets within a minimum size support for a given
    number of vanishing moments, but they are as symmetrical as possible,
    as opposed to the Daubechies filters which are highly asymmetrical.
    They are indexed by Par, which specifies the number of vanishing
    moments and is equal to half the size of the support. It ranges
    from 4 to 10.

    The Vaidyanathan filter gives an exact reconstruction, but does not
    satisfy any moment condition.  The filter has been optimized for
    speech coding.

    The Battle-Lemarie filter generate spline orthogonal wavelet basis.
    The parameter Par gives the degree of the spline. The number of
    vanishing moments is Par+1.

    See Also: FWT_PO, IWT_PO, FWT2_PO, IWT2_PO, WPAnalysis

    References: The books by Daubechies and Wickerhauser.

    Part of  WaveLab850 (http://www-stat.stanford.edu/~wavelab/)
    """
    if Type == 'Haar':
        onFilter = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    if Type == 'Beylkin':
        onFilter = np.array([.099305765374, .424215360813, .699825214057,
                    .449718251149, -.110927598348, -.264497231446,
                    .026900308804, .155538731877, -.017520746267,
                    -.088543630623, .019679866044, .042916387274,
                    -.017460408696, -.014365807969, .010040411845,
                    .001484234782, -.002736031626, .000640485329])
    if Type == 'Coiflet':
        if Par == 1:
            onFilter = np.array([.038580777748, -.126969125396, -.077161555496,
                                .607491641386, .745687558934, .226584265197])
        elif Par == 2:
            onFilter = np.array([.016387336463, -.041464936782, -.067372554722,
                                .386110066823, .812723635450, .417005184424,
                                -.076488599078, -.059434418646, .023680171947,
                                .005611434819, -.001823208871, -.000720549445])
        elif Par == 3:
            onFilter = np.array([-.003793512864, .007782596426, .023452696142,
                                -.065771911281,	-.061123390003,	.405176902410,
				                .793777222626,	.428483476378,	-.071799821619,
				                -.082301927106,	.034555027573,	.015880544864,
				                -.009007976137,	-.002574517688,	.001117518771,
				                .000466216960,	-.000070983303,	-.000034599773])
        elif Par == 4:
            onFilter = np.array([.000892313668,	-.001629492013,	-.007346166328,
				                .016068943964,	.026682300156,	-.081266699680,
				                -.056077313316,	.415308407030,	.782238930920,
				                .434386056491,	-.066627474263,	-.096220442034,
				                .039334427123,	.025082261845,	-.015211731527,
				                -.005658286686,	.003751436157,	.001266561929,
				                -.000589020757,	-.000259974552,	.000062339034,
				                .000031229876,	-.000003259680,	-.000001784985])
        elif Par == 5:
            onFilter = np.array([-.000212080863, .000358589677,	.002178236305,
				                -.004159358782,	-.010131117538,	.023408156762,
				                .028168029062,	-.091920010549,	-.052043163216,
				                .421566206729,	.774289603740,	.437991626228,
				                -.062035963906,	-.105574208706,	.041289208741,
				                .032683574283,	-.019761779012,	-.009164231153,
				                .006764185419,	.002433373209,	-.001662863769,
				                -.000638131296,	.000302259520,	.000140541149,
				                -.000041340484,	-.000021315014,	.000003734597,
				                .000002063806,	-.000000167408,	-.000000095158])
    if Type == 'Daubechies':
        if Par == 4:
            onFilter = np.array([.482962913145,	.836516303738, .224143868042,
            	                   -.129409522551])
        elif Par == 6:
            onFilter = np.array([.332670552950,	.806891509311, .459877502118,
            	                 -.135011020010, -.085441273882, .035226291882])
        elif Par == 8:
            onFilter = np.array([.230377813309,	.714846570553, .630880767930,
            	               -.027983769417, -.187034811719, .030841381836,
				                .032883011667, -.010597401785])
        elif Par == 10:
            onFilter = np.array([.160102397974,	.603829269797,	.724308528438,
				                .138428145901,	-.242294887066,	-.032244869585,
				                .077571493840,	-.006241490213,	-.012580751999,
				                .003335725285])
        elif Par == 12:
            onFilter = np.array([.111540743350,	.494623890398, .751133908021,
				                .315250351709, -.226264693965, -.129766867567,
				                .097501605587, .027522865530, -.031582039317,
				                .000553842201, .004777257511, -.001077301085])
        elif Par == 14:
            onFilter = np.array([.077852054085, .396539319482, .729132090846,
				                .469782287405, -.143906003929, -.224036184994,
				                .071309219267, .080612609151, -.038029936935,
				                -.016574541631, .012550998556, .000429577973,
				                -.001801640704, .000353713800])
        elif Par == 16:
            onFilter = np.array([.054415842243, .312871590914, .675630736297,
				                .585354683654, -.015829105256, -.284015542962,
				                .000472484574, .128747426620, -.017369301002,
                                -.044088253931, .013981027917, .008746094047,
                                -.004870352993, -.000391740373, .000675449406,
                                -.000117476784])
        elif Par==18:
            onFilter = np.array([.038077947364, .243834674613, .604823123690,
            			         .657288078051, .133197385825, -.293273783279,
                                 -.096840783223, .148540749338, .030725681479,
                                 -.067632829061, .000250947115, .022361662124,
                                 -.004723204758, -.004281503682, .001847646883,
                                 .000230385764, -.000251963189, .000039347320])
        elif Par==20:
            onFilter = np.array([.026670057901, .188176800078, .527201188932,
				                .688459039454, .281172343661, -.249846424327,
                                -.195946274377,	.127369340336,	.093057364604,
                                -.071394147166,	-.029457536822,	.033212674059,
                                .003606553567,	-.010733175483,	.001395351747,
                                .001992405295,	-.000685856695,	-.000116466855,
                                .000093588670,	-.000013264203])
    if Type == 'Symmlet':
        if Par == 4:
            onFilter = np.array([-.107148901418, -.041910965125, .703739068656,
                                1.136658243408, .421234534204, -.140317624179,
                                -.017824701442,	.045570345896])
        elif Par == 5:
            onFilter = np.array([.038654795955, .041746864422, -.055344186117,
                                .281990696854, 1.023052966894, .896581648380,
                                .023478923136,	-.247951362613,	-.029842499869,
                                .027632152958])
        elif Par == 6:
            onFilter = np.array([.021784700327, .004936612372, -.166863215412,
				                -.068323121587, .694457972958, 1.113892783926,
                                .477904371333, -.102724969862, -.029783751299,
                                .063250562660, .002499922093, -.011031867509])
        elif Par == 7:
            onFilter = np.array([.003792658534, -.001481225915, -.017870431651,
                                .043155452582, .096014767936, -.070078291222,
                                .024665659489, .758162601964, 1.085782709814,
                                .408183939725, -.198056706807, -.152463871896,
                                .005671342686, .014521394762])
        elif Par == 8:
            onFilter = np.array([.002672793393, -.000428394300,	-.021145686528,
                                .005386388754, .069490465911, -.038493521263,
                                -.073462508761, .515398670374, 1.099106630537,
                                .680745347190, -.086653615406, -.202648655286,
                                .010758611751, .044823623042, -.000766690896,
                                -.004783458512])
        elif Par == 9:
            onFilter = np.array([.001512487309, -.000669141509, -.014515578553,
                                .012528896242, .087791251554, -.025786445930,
                                -.270893783503, .049882830959, .873048407349,
                                1.015259790832,	.337658923602, -.077172161097,
                                .000825140929, .042744433602, -.016303351226,
                                -.018769396836, .000876502539, .001981193736])
        elif Par == 10:
            onFilter = np.array([.001089170447, .000135245020, -.012220642630,
                                -.002072363923,	.064950924579, .016418869426,
                                -.225558972234,	-.100240215031, .667071338154,
                                1.088251530500,	.542813011213, -.050256540092,
                                -.045240772218, .070703567550, .008152816799,
                                -.028786231926, -.001137535314, .006495728375,
                                .000080661204, -.000649589896])
    if Type == 'Vaidyanathan':
        onFilter = np.array([-.000062906118, .000343631905, -.000453956620,
			                 -.000944897136, .002843834547, .000708137504,
                             -.008839103409, .003153847056, .019687215010,
                             -.014853448005, -.035470398607, .038742619293,
                             .055892523691,	-.077709750902,	-.083928884366,
                             .131971661417, .135084227129, -.194450471766,
                             -.263494802488, .201612161775, .635601059872,
                             .572797793211, .250184129505, .045799334111])
    if Type == 'Battle':
        if Par == 1:
            onFilterTmp = np.array([0.578163, 0.280931, -0.0488618, -0.0367309,
                                    0.012003, 0.00706442, -0.00274588,
                                    -0.00155701, 0.000652922, 0.000361781,
                                    -0.000158601, -0.0000867523])
        elif Par == 3:
            onFilterTmp = np.array([0.541736, 0.30683, -0.035498, -0.0778079,
                                    0.0226846, 0.0297468, -0.0121455,
                                    -0.0127154, 0.00614143, 0.00579932,
                                    -0.00307863, -0.00274529, 0.00154624,
                                    0.00133086, -0.000780468, -0.00065562,
                                    0.000395946, 0.000326749, -0.000201818,
                                    -0.000164264, 0.000103307])
        elif Par == 5:
            onFilterTmp = np.array([0.528374, 0.312869, -0.0261771, -0.0914068,
                                   0.0208414, 0.0433544, -0.0148537, -0.0229951,
                                0.00990635, 0.0128754, -0.00639886, -0.00746848,
                               0.00407882, 0.00444002, -0.00258816, -0.00268646,
                               0.00164132, 0.00164659, -0.00104207, -0.00101912,
                           0.000662836, 0.000635563, -0.000422485, -0.000398759,
                           0.000269842, 0.000251419, -0.000172685, -0.000159168,
                           0.000110709, 0.000101113])
        onFilter = np.zeros(2*onFilterTmp.size-1)
        onFilter[onFilterTmp.size-1:2*onFilterTmp.size] = onFilterTmp;
        onFilter[0:onFilterTmp.size-1] = onFilterTmp[onFilterTmp.size-1:0:-1]
    return onFilter / np.linalg.norm(onFilter)

"""
 Copyright (c) 1993-5. Jonathan Buckheit and David Donoho

  Part of Wavelab Version 850
  Built Tue Jan  3 13:20:40 EST 2006
  This is Copyrighted Material
  For Copying permissions see COPYING.m
  Comments? e-mail wavelab@stat.stanford.edu
"""


def dfilters(fname, type):
    """
    This is a translation of the original Matlab implementation of dfilters.m
    from the Nonsubsampled Contourlet Toolbox. The following comment is from
    the original and only applies in so far that not all of the directional
    filters are implemented in this Python version but only those which are
    needed for the shearlet toolbox.

    DFILTERS	Generate directional 2D filters

    	[h0, h1] = dfilters(fname, type)

    Input:

    	fname:	Filter name.  Available 'fname' are:
    		'haar':	the "Haar" filters
    		'vk':	McClellan transformed of the filter
                    from the VK book
    		'ko':	orthogonal filter in the Kovacevics
                    paper
    		'kos':	smooth 'ko' filter
    		'lax':	17 x 17 by Lu, Antoniou and Xu
    		'sk':	9 x 9 by Shah and Kalker
    		'cd':	7 and 9 McClellan transformed by
    				Cohen and Daubechies
    		'pkva':	ladder filters by Phong et al.
    		'oqf_362': regular 3 x 6 filter
           'dvmlp': regular linear phase biorthogonal filter
                    with 3 dvm
    		'sinc':	ideal filter (*NO perfect recontruction*)
           'dmaxflat': diamond maxflat filters obtained from a three
                        stage ladder

    	     type:	'd' or 'r' for decomposition or reconstruction filters

     Output:
    	h0, h1:	diamond filter pair (lowpass and highpass)

     To test those filters (for the PR condition for the FIR case), verify that:
     conv2(h0, modulate2(h1, 'b')) + conv2(modulate2(h0, 'b'), h1) = 2
     (replace + with - for even size filters)

     To test for orthogonal filter
     conv2(h, reverse2(h)) + modulate2(conv2(h, reverse2(h)), 'b') = 2

     Part of the Nonsubsampled Contourlet Toolbox
     (http://www.mathworks.de/matlabcentral/fileexchange/10049-nonsubsampled-contourlet-toolbox)
    """
    if fname == 'haar':
        if type.lower() == 'd':
            h0 = np.array([1, 1]) / np.sqrt(2)
            h1 = np.array([-1, 1]) / np.sqrt(2)
        else:
            h0 = np.array([1, 1]) / np.sqrt(2)
            h1 = np.array([1, -1]) / np.sqrt(2)
    elif fname == 'vk':                         # in Vetterli and Kovacevic book
        if type.lower() == 'd':
            h0 = np.array([1, 2, 1]) / 4
            h1 = np.array([-1, -2, 6, -2, -1]) / 4
        else:
            h0 = np.array([-1, 2, 6, 2, -1]) / 4
            h1 = np.array([-1, 2, -1]) / 4
        t = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4      # diamon kernel
        h0 = mctrans(h0, t)
        h1 = mctrans(h1, t)
    elif fname == 'ko':                # orthogonal filters in Kovacevics thesis
        a0 = 2
        a1 = 0.5
        a2 = 1
        h0 = np.array([[0,    -a1,  -a0*a1, 0],
                        [-a2, -a0*a2, -a0,  1],
                        [0, a0*a1*a2, -a1*a2, 0]])
        # h1 = qmf2(h0)
        h1 = np.array([[0, -a1*a2, -a0*a1*a2, 0],
                       [1,   a0,  -a0*a2,  a2],
                       [0, -a0*a1,   a1,   0]])
        # normalize filter sum and norm
        norm = np.sqrt(2) / np.sum(h0)
        h0 = h0 * norm
        h1 = h1 * norm

        if type == 'r':
            # reverse filters for reconstruction
            h0 = h0[::-1]
            h1 = h1[::-1]
    elif fname == 'kos':        # smooth orthogonal filters in Kovacevics thesis
        a0 = -np.sqrt(3)
        a1 = -np.sqrt(3)
        a2 = 2+np.sqrt(3)

        h0 = np.array([[0,    -a1,  -a0*a1, 0],
                        [-a2, -a0*a2, -a0,  1],
                        [0, a0*a1*a2, -a1*a2, 0]])
        # h1 = qmf2(h0)
        h1 = np.array([[0, -a1*a2, -a0*a1*a2, 0],
                       [1,   a0,  -a0*a2,  a2],
                       [0, -a0*a1,   a1,   0]])
        # normalize filter sum and norm
        norm = np.sqrt(2) / np.sum(h0)
        h0 = h0 * norm
        h1 = h1 * norm

        if type == 'r':
            # reverse filters for reconstruction
            h0 = h0[::-1]
            h1 = h1[::-1]
    elif fname == 'lax':                                # by lu, antoniou and xu
        h = np.array([[-1.2972901e-5,  1.2316237e-4, -7.5212207e-5,  6.3686104e-5,
                    9.4800610e-5, -7.5862919e-5,  2.9586164e-4, -1.8430337e-4],
                    [1.2355540e-4, -1.2780882e-4, -1.9663685e-5, -4.5956538e-5,
                    -6.5195193e-4, -2.4722942e-4, -2.1538331e-5, -7.0882131e-4],
                    [-7.5319075e-5, -1.9350810e-5, -7.1947086e-4,  1.2295412e-3,
                    5.7411214e-4,  4.4705422e-4,  1.9623554e-3,  3.3596717e-4],
                    [6.3400249e-5, -2.4947178e-4,  4.4905711e-4, -4.1053629e-3,
                    -2.8588307e-3,  4.3782726e-3, -3.1690509e-3, -3.4371484e-3],
                    [9.6404973e-5, -4.6116254e-5,  1.2371871e-3, -1.1675575e-2,
                    1.6173911e-2, -4.1197559e-3,  4.4911165e-3,  1.1635130e-2],
                    [-7.6955555e-5, -6.5618379e-4,  5.7752252e-4,  1.6211426e-2,
                    2.1310378e-2, -2.8712621e-3, -4.8422645e-2, -5.9246338e-3],
                    [2.9802986e-4, -2.1365364e-5,  1.9701350e-3,  4.5047673e-3,
                    -4.8489158e-2, -3.1809526e-3, -2.9406153e-2,  1.8993868e-1],
                    [-1.8556637e-4, -7.1279432e-4,  3.3839195e-4,  1.1662001e-2,
                    -5.9398223e-3, -3.4467920e-3,  1.9006499e-1,  5.7235228e-1]
                    ])
        h0 = np.sqrt(2) * np.append(h, h[:,-2::-1], 1)
        h0 = np.append(h0, h0[-2::-1,:], 0)
        h1 = modulate2(h0, 'b')
    elif fname == 'sk':                                 # by shah and kalker
        h = np.array([[0.621729, 0.161889, -0.0126949, -0.00542504, 0.00124838],
                    [0.161889, -0.0353769, -0.0162751, -0.00499353, 0],
                    [-0.0126949,  -0.0162751,  0.00749029, 0, 0],
                    [-0.00542504, 0.00499353, 0, 0, 0],
                    [0.00124838, 0, 0, 0, 0]])
        h0 = np.append(h[-1:0:-1, -1:0:-1], h[-1:0:-1,:], 1)
        h0 = np.append(h0, np.append(h[:,-1:0:-1], h, 1), 0)*np.sqrt(2)
        h1 = modulate2(h0, 'b')
    elif fname == 'dvmlp':
            q = np.sqrt(2)
            b = 0.02
            b1 = b*b;
            h  = np.array([[b/q, 0, -2*q*b, 0, 3*q*b, 0, -2*q*b, 0, b/q],
                [0, -1/(16*q), 0, 9/(16*q), 1/q, 9/(16*q), 0, -1/(16*q), 0],
                [b/q, 0, -2*q*b, 0, 3*q*b, 0, -2*q*b, 0, b/q]])
            g0 = np.array([[-b1/q, 0, 4*b1*q, 0, -14*q*b1, 0, 28*q*b1, 0,
                    -35*q*b1, 0, 28*q*b1, 0, -14*q*b1, 0, 4*b1*q, 0, -b1/q],
                [0, b/(8*q), 0, -13*b/(8*q), b/q, 33*b/(8*q), -2*q*b,
                        -21*b/(8*q), 3*q*b, -21*b/(8*q), -2*q*b, 33*b/(8*q),
                        b/q, -13*b/(8*q), 0, b/(8*q), 0],
                [-q*b1, 0, -1/(256*q) + 8*q*b1, 0, 9/(128*q) - 28*q*b1,
                    -1/(q*16), -63/(256*q) + 56*q*b1, 9/(16*q),
                    87/(64*q)-70*q*b1, 9/(16*q), -63/(256*q) + 56*q*b1,
                    -1/(q*16), 9/(128*q) - 28*q*b1, 0, -1/(256*q) + 8*q*b1, 0,
                    -q*b1],
                [0, b/(8*q), 0, -13*b/(8*q), b/q, 33*b/(8*q), -2*q*b,
                    -21*b/(8*q), 3*q*b, -21*b/(8*q), -2*q*b, 33*b/(8*q), b/q,
                    -13*b/(8*q), 0, b/(8*q), 0],
                [-b1/q, 0, 4*b1*q, 0, -14*q*b1, 0, 28*q*b1, 0, -35*q*b1, 0,
                    28*q*b1, 0, -14*q*b1, 0, 4*b1*q, 0, -b1/q]])
            h1 = modulate2(g0, 'b')
            h0 = h
            print(h1.shape)
            print(h0.shape)
            if type == 'r':
                h1 = modulate2(h, 'b')
                h0 = g0
    elif fname == 'cd' or fname == '7-9':        # by cohen and Daubechies
        h0 = np.array([0.026748757411, -0.016864118443, -0.078223266529,
                    0.266864118443, 0.602949018236, 0.266864118443,
                    -0.078223266529, -0.016864118443, 0.026748757411])
        g0 = np.array([-0.045635881557, -0.028771763114, 0.295635881557,
	                   0.557543526229, 0.295635881557, -0.028771763114,
	                    -0.045635881557])
        if type == 'd':
            h1 = modulate2(g0, 'c')
        else:
            h1 = modulate2(h0, 'c')
            h0 = g0
        # use McClellan to obtain 2D filters
        t = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])/4 # diamond kernel
        h0 = np.sqrt(2) * mctrans(h0, t)
        h1 = np.sqrt(2) * mctrans(h1, t)
    elif fname == 'oqf_362':
        h0 = np.sqrt(2) / 64 * np.array([[np.sqrt(15), -3, 0],
                        [0, 5, np.sqrt(15)], [-2*np.sqrt(2), 30, 0],
                        [0, 30, 2*np.sqrt(15)], [np.sqrt(15), 5, 0],
                        [0, -3, -np.sqrt(15)]])
        h1 = -modulate2(h0, 'b')
        h1 = -h1[::-1]
        if type == 'r':
            h0 = h0[::-1]
            h1 = -modulate2(h0, 'b')
            h1 = -h1[::-1]
    elif fname == 'test':
        h0 = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]])
        h1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    elif fname == 'testDVM':
        h0 = np.array([[1, 1], [1, 1]]) / np.sqrt(2)
        h1 = np.array([[-1, 1], [1, -1]]) / np.sqrt(2)
    elif fname == 'qmf':                # by Lu, antoniou and xu
        # ideal response window
        m = 2
        n = 2
        w1d = np.kaiser(4*m+1, 2.6)
        w = np.zeros((n+m+1,n+m+1))
        for n1 in np.arange(-m,m+1):
            for n2 in np.arange(-n,n+1):
                w[n1+m,n2+n] = w1d[2*m+n1+n2]*w1d[2*m+n1-n2]
        h = np.zeros((n+m+1,n+m+1))
        for n1 in np.arange(-m,m+1):
            for n2 in np.arange(-n,n+1):
                h[n1+m, n2+n] = 0.5*np.sinc((n1+n2)/2) * 0.5*np.sinc((n1-n2)/2)
        c = np.sum(h)
        h = np.sqrt(2) * h
        h0 = h * w
        h1 = modulate2(h0, 'b')
    elif fname == 'qmf2':                       # by Lu, Antoniou and Xu
	   # ideal response window
        h = np.array([
            [-0.001104, 0.002494, -0.001744, 0.004895, -0.000048, -0.000311],
             [0.008918, -0.002844, -0.025197, -0.017135, 0.003905, -0.000081],
             [-0.007587, -0.065904, 00.100431, -0.055878, 0.007023, 0.001504],
             [0.001725, 0.184162, 0.632115, 0.099414, -0.027006, -0.001110],
             [-0.017935, -0.000491, 0.191397, -0.001787, -0.010587, 0.002060],
             [0.001353, 0.005635, -0.001231, -0.009052, -0.002668, 0.000596]])
        h0 = h/np.sum(h)
        h1 = modulate2(h0, 'b')
    elif fname == 'dmaxflat4':
        M1 = 1/np.sqrt(2)
        M2 = np.copy(M1)
        k1 = 1-np.sqrt(2)
        k3 = np.copy(k1)
        k2 = np.copy(M1)
        h = np.array([0.25*k2*k3, 0.5*k2, 1+0.5*k2*k3])*M1
        h = np.append(h, h[-2::-1])
        g = np.array([-0.125*k1*k2*k3, 0.25*k1*k2,
                    -0.5*k1-0.5*k3-0.375*k1*k2*k3, 1+0.5*k1*k2])*M2
        g = np.append(g, h[-2::-1])
        B = dmaxflat(4,0)
        h0 = mctrans(h,B)
        g0 = mctrans(g,B)
        h0 = np.sqrt(2) * h0 / np.sum(h0)
        g0 = np.sqrt(2) * g0 / np.sum(g0)

        h1 = modulate2(g0, 'b')
        if type == 'r':
            h1 = modulate2(h0, 'b')
            h0 = g0
    elif fname == 'dmaxflat5':
        M1 = 1/np.sqrt(2)
        M2 = M1
        k1 = 1-np.sqrt(2)
        k3 = k1
        k2 = M1
        h = np.array([0.25*k2*k3, 0.5*k2, 1+0.5*k2*k3])*M1
        h = np.append(h, h[-2::-1])
        g = np.array([-0.125*k1*k2*k3, 0.25*k1*k2,
                    -0.5*k1-0.5*k3-0.375*k1*k2*k3, 1+0.5*k1*k2])*M2
        g = np.append(g, h[-2::-1])
        B = dmaxflat(5,0)
        h0 = mctrans(h,B)
        g0 = mctrans(g,B)
        h0 = np.sqrt(2) * h0 / np.sum(h0)
        g0 = np.sqrt(2) * g0 / np.sum(g0)

        h1 = modulate2(g0, 'b')
        if type == 'r':
            h1 = modulate2(h0, 'b')
            h0 = g0
    elif fname == 'dmaxflat6':
        M1 = 1/np.sqrt(2)
        M2 = M1
        k1 = 1-np.sqrt(2)
        k3 = k1
        k2 = M1
        h = np.array([0.25*k2*k3, 0.5*k2, 1+0.5*k2*k3])*M1
        h = np.append(h, h[-2::-1])
        g = np.array([-0.125*k1*k2*k3, 0.25*k1*k2,
                    -0.5*k1-0.5*k3-0.375*k1*k2*k3, 1+0.5*k1*k2])*M2
        g = np.append(g, h[-2::-1])
        B = dmaxflat(6,0)
        h0 = mctrans(h,B)
        g0 = mctrans(g,B)
        h0 = np.sqrt(2) * h0 / np.sum(h0)
        g0 = np.sqrt(2) * g0 / np.sum(g0)

        h1 = modulate2(g0, 'b')
        if type == 'r':
            h1 = modulate2(h0, 'b')
            h0 = g0
    elif fname == 'dmaxflat7':
        M1 = 1/np.sqrt(2)
        M2 = M1
        k1 = 1-np.sqrt(2)
        k3 = k1
        k2 = M1
        h = np.array([0.25*k2*k3, 0.5*k2, 1+0.5*k2*k3])*M1
        h = np.append(h, h[-2::-1])
        g = np.array([-0.125*k1*k2*k3, 0.25*k1*k2,
                    -0.5*k1-0.5*k3-0.375*k1*k2*k3, 1+0.5*k1*k2])*M2
        g = np.append(g, h[-2::-1])
        B = dmaxflat(7,0)
        h0 = mctrans(h,B)
        g0 = mctrans(g,B)
        h0 = np.sqrt(2) * h0 / np.sum(h0)
        g0 = np.sqrt(2) * g0 / np.sum(g0)

        h1 = modulate2(g0, 'b')
        if type == 'r':
            h1 = modulate2(h0, 'b')
            h0 = g0
    # The original file supports a case "otherwise" for unrecognized filters
    # and computes simple 1D wavelet filters for them using wfilters.m
    # I think we don't need this and skip this for the time being.
    # IN ORIGINAL MATLAB VERSION:
    # otherwise
    # % Assume the "degenerated" case: 1D wavelet filters
    # [h0,h1] = wfilters(fname, type);
    return h0, h1


def dmaxflat(N,d):
    """
    THIS IS A REWRITE OF THE ORIGINAL MATLAB IMPLEMENTATION OF dmaxflat.m
    FROM THE Nonsubsampled Contourlet Toolbox.   -- Stefan Loock, Dec 2016.

    returns 2-D diamond maxflat filters of order 'N'
    the filters are nonseparable and 'd' is the (0,0) coefficient, being 1 or 0
    depending on use.
    by Arthur L. da Cunha, University of Illinois Urbana-Champaign
    Aug 2004
    """
    if (N > 7) or (N < 1):
        print('Error: N must be in {1,2,...,7}')
        return 0
    if N == 1:
        h = np.array([[0, 1, 0],[1, 0, 1],[0, 1, 0]])/4
        h[1,1] = d
    elif N == 2:
        h = np.array([[0, -1, 0],[-1, 0, 10], [0, 10, 0]])
        h = np.append(h, np.fliplr(h[:,0:-1]), 1)
        h = np.append(h, np.flipud(h[0:-1,:]), 0)/32
        h[2,2] = d
    elif N == 3:
        h = np.array([[0, 3, 0, 2],[3, 0, -27, 0],[0, -27, 0, 174],
                        [2, 0, 174, 0]])
        h = np.append(h, np.fliplr(h[:, 0:-1]), 1)
        h = np.append(h, np.flipud(h[0:-1,:]),0)
        h[3,3] = d
    elif N == 4:
        h = np.array([[0, -5, 0, -3, 0], [-5, 0, 52, 0, 34],
                        [0, 52, 0, -276, 0], [-3, 0, -276, 0, 1454],
                        [0, 34, 0, 1454, 0]])/np.power(2,12)
        h = np.append(h, np.fliplr(h[:,0:-1]),1)
        h = np.append(h, np.flipud(h[0:-1,:]),0)
        h[4,4] = d
    elif N == 5:
        h = np.array([[0, 35, 0, 20, 0, 18], [35, 0, -425, 0, -250, 0],
                    [0, -425, 0, 2500, 0, 1610], [20, 0, 2500, 0, -10200, 0],
                    [0, -250, 0, -10200, 0, 47780],
                    [18, 0, 1610, 0, 47780, 0]])/np.power(2,17)
        h = np.append(h, np.fliplr(h[:,0:-1]),1)
        h = np.append(h, np.flipud(h[0:-1,:]),0)
        h[5,5] = d
    elif N == 6:
        h = np.array([[0, -63, 0, -35, 0, -30, 0],
                     [-63, 0, 882, 0, 495, 0, 444],
                     [0, 882, 0, -5910, 0, -3420, 0],
                     [-35, 0, -5910, 0, 25875, 0, 16460],
                     [0, 495, 0, 25875, 0, -89730, 0],
                     [-30, 0, -3420, 0, -89730, 0, 389112],
                     [0, 44, 0, 16460, 0, 389112, 0]])/np.power(2,20)
        h = np.append(h, np.fliplr(h[:,0:-1]),1)
        h = np.append(h, np.flipud(h[0:-1,:]),0)
        h[6,6] = d
    elif N == 7:
        h = np.array([[0, 231, 0, 126, 0, 105, 0, 100],
                    [231, 0, -3675, 0, -2009, 0, -1715, 0],
                    [0, -3675, 0, 27930, 0, 15435, 0, 13804],
                    [126, 0, 27930, 0, -136514, 0, -77910, 0],
                    [0, -2009, 0, -136514, 0, 495145, 0, 311780],
                    [105, 0, 15435, 0, 495145, 0, -1535709, 0],
                    [0, -1715, 0, -77910, 0, -1534709, 0, 6305740],
                    [100, 0, 13804, 0, 311780, 0, 6305740, 0]])/np.power(2,24)
        h = np.append(h, np.fliplr(h[:,0:-1]),1)
        h = np.append(h, np.flipud(h[0:-1,:]),0)
        h[7,7] = d
    return h


def mctrans(b,t):
    """
    This is a translation of the original Matlab implementation of mctrans.m
    from the Nonsubsampled Contourlet Toolbox by Arthur L. da Cunha.

    MCTRANS McClellan transformation

        H = mctrans(B,T)

    produces the 2-D FIR filter H that corresponds to the 1-D FIR filter B
    using the transform T.


    Convert the 1-D filter b to SUM_n a(n) cos(wn) form

    Part of the Nonsubsampled Contourlet Toolbox
    (http://www.mathworks.de/matlabcentral/fileexchange/10049-nonsubsampled-contourlet-toolbox)
    """

    # Convert the 1-D filter b to SUM_n a(n) cos(wn) form
    # if mod(n,2) != 0 -> error
    n = (b.size-1)//2

    b = fftlib.fftshift(b[::-1]) #inverse fftshift
    b = b[::-1]
    a = np.zeros(n+1)
    a[0] = b[0]
    a[1:n+1] = 2*b[1:n+1]

    inset = np.floor((np.asarray(t.shape)-1)/2)
    inset = inset.astype(int)
    # Use Chebyshev polynomials to compute h
    P0 = 1
    P1 = t;
    h = a[1]*P1;
    rows = int(inset[0]+1)
    cols = int(inset[1]+1)
    h[rows-1,cols-1] = h[rows-1,cols-1]+a[0]*P0;
    for i in range(3,n+2):
        P2 = 2*signal.convolve2d(t, P1)
        rows = (rows + inset[0]).astype(int)
        cols = (cols + inset[1]).astype(int)
        if i == 3:
            P2[rows-1,cols-1] = P2[rows-1,cols-1] - P0
        else:
            P2[rows[0]-1:rows[-1],cols[0]-1:cols[-1]] = P2[rows[0]-1:rows[-1],
                                                        cols[0]-1:cols[-1]] - P0
        rows = inset[0] + np.arange(np.asarray(P1.shape)[0])+1
        rows = rows.astype(int)
        cols = inset[1] + np.arange(np.asarray(P1.shape)[1])+1
        cols = cols.astype(int)
        hh = h;
        h = a[i-1]*P2
        h[rows[0]-1:rows[-1], cols[0]-1:cols[-1]] = h[rows[0]-1:rows[-1],
                                                        cols[0]-1:cols[-1]] + hh
        P0 = P1;
        P1 = P2;
    h = np.rot90(h,2)
    return h


def modulate2(x, type, center=np.array([0, 0])):
    """
    THIS IS A REWRITE OF THE ORIGINAL MATLAB IMPLEMENTATION OF
    modulate2.m FROM THE Nonsubsampled Contourlet Toolbox.

    MODULATE2	2D modulation

            y = modulate2(x, type, [center])

    With TYPE = {'r', 'c' or 'b'} for modulate along the row, or column or
    both directions.

    CENTER secify the origin of modulation as floor(size(x)/2)+1+center
    (default is [0, 0])

    Part of the Nonsubsampled Contourlet Toolbox
    (http://www.mathworks.de/matlabcentral/fileexchange/10049-nonsubsampled-contourlet-toolbox)
    """
    size = np.asarray(x.shape)
    if x.ndim == 1:
        if np.array_equal(center, [0, 0]):
            center = 0
    origin = np.floor(size/2)+1+center
    n1 = np.arange(size[0])-origin[0]+1
    if x.ndim == 2:
        n2 = np.arange(size[1])-origin[1]+1
    else:
        n2 = n1
    if type == 'r':
        m1 = np.power(-1,n1)
        if x.ndim == 1:
            y = x*m1
        else:
            y = x * np.transpose(np.tile(m1, (size[1], 1)))
    elif type == 'c':
        m2 = np.power(-1,n2)
        if x.ndim == 1:
            y = x*m2
        else:
            y = x * np.tile(m2, np.array([size[0], 1]))
    elif type == 'b':
        m1 = np.power(-1,n1)
        m2 = np.power(-1,n2)
        m = np.outer(m1, m2)
        if x.ndim == 1:
            y = x * m1
        else:
            y = x * m
    return y

def MirrorFilt(x):
    """
    This is a translation of the original Matlab implementation of
    MirrorFilt.m from the WaveLab850 toolbox.

     MirrorFilt -- Apply (-1)^t modulation
      Usage

            h = MirrorFilt(l)

      Inputs

            l   1-d signal

      Outputs

            h   1-d signal with DC frequency content shifted
                to Nyquist frequency

      Description

            h(t) = (-1)^(t-1)  * x(t),  1 <= t <= length(x)

      See Also: DyadDownHi

    Part of  WaveLab850 (http://www-stat.stanford.edu/~wavelab/)
    """
    return np.power(-1,np.arange(x.size))*x

    """
    Copyright (c) 1993. Iain M. Johnstone

    Part of Wavelab Version 850
    Built Tue Jan  3 13:20:40 EST 2006
    This is Copyrighted Material
    For Copying permissions see COPYING.m
    Comments? e-mail wavelab@stat.stanford.edu
    """
