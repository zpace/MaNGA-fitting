import numpy as np
import matplotlib.pyplot as plt

import astropy.io.fits as fits
import astropy.table as table

from astropy import units as u, constants as c
import ppxf_util as util
from scipy import ndimage
import manga_tools as m


def convolve_variable_width(a, sig, prec=1.):
    '''
    approximate convolution with a kernel that varies along the spectral
        direction, by stretching the data by the inverse of the kernel's
        width at a given position

    N.B.: this is an approximation to the proper operation, which
        involves iterating over each pixel of each template and
        performing ~10^6 convolution operations

    Parameters:
     - a: N-D array; convolution will occur along the final axis
     - sig: 1-D array (must have same length as the final axis of a);
        describes the varying width of the kernel
     - prec: precision argument. When higher, oversampling is more thorough
    '''

    assert (len(sig) == a.shape[-1]), '\tLast dimension of `a` must equal \
        length of `sig` (each element of a must have a convolution width)'

    sig0 = sig.max()  # the "base" width that results in minimal blurring
    # if prec = 1, just use sig0 as base.
    n = np.rint(prec * sig0/sig).astype(int)
    # print n
    print '\tWarped array length: {}'.format(n.sum())
    # define "warped" array a_w with n[i] instances of a[:,:,i]
    a_w = np.repeat(a, n, axis=-1)
    # now a "warped" array sig_w
    sig_w = np.repeat(sig, n)

    # define start and endpoints for each value
    nl = np.cumsum(np.insert(n, 0, 0))[:-1]
    nr = np.cumsum(n)
    # now the middle of the interval
    nm = np.rint(np.median(np.column_stack((nl, nr)), axis=1)).astype(int)

    # print nm

    # print a_w.shape, sig_w.shape # check against n.sum()

    # now convolve the whole thing with a Gaussian of width sig0
    print '\tCONVOLVE...'
    # account for the increased precision required
    a_w_f = np.empty_like(a_w)
    # have to iterate over the rows and columns, to avoid MemoryError
    c = 0  # counter (don't judge me, it was early in the morning)
    for i in range(a_w_f.shape[0]):
        for j in range(a_w_f.shape[1]):
            c += 1
            print '\t\tComputing convolution {} of {}...'.format(
                c, a_w_f.shape[0] * a_w_f.shape[1])
            a_w_f[i, j, :] = ndimage.gaussian_filter1d(
                a_w[i, j, :], prec*sig0)
    # print a_w_f.shape # should be the same as the original shape

    # and downselect the pixels (take approximate middle of each slice)
    # f is a mask that will be built and applied to a_w_f

    # un-warp the newly-convolved array by selecting only the slices
    # in dimension -1 that are in nm
    a_f = a_w_f[:, :, nm]

    return a_f


def setup_MaNGA_stellar_libraries(fname_ifu, fname_tem, plot=False):
    '''
    set up all the required stellar libraries for a MaNGA datacube

    this should only need to be run once.
    '''

    print 'Reading drpall...'
    drpall = fits.open(m.drpall_loc + 'drpall-v1_3_3.fits')[0].data

    print 'Reading SSPs...'
    MaNGA_hdu = fits.open(fname_ifu)

    # open global MaNGA header
    glob_h = MaNGA_hdu[0].header

    print 'Constructing wavelength grid...'
    # read in some average value for wavelength solution and spectral res
    L_ifu = m.wave(MaNGA_hdu).data
    R_avg, l_avg, = m.res_over_plate('MPL-3', '7443', plot=plot)
    FWHM_avg = l_avg / R_avg  # FWHM of a galaxy in AA at some wavelength

    # now read in basic info about templates and
    # up-sample "real" spectral resolution to the model wavelength grid
    tems = fits.open(fname_tem)[0]
    htems = tems.header
    logL_tem = np.linspace(
        htems['CRVAL3'],
        htems['CRVAL3'] + tems.data.shape[2]*htems['CDELT3'],
        tems.data.shape[2])  # base e
    L_tem = np.exp(logL_tem)

    dL_tem = np.empty_like(L_tem)
    dL_tem[:-1] = L_tem[1:] - L_tem[:-1]
    dL_tem[-1] = dL_tem[-2]  # this is not exact, but it's efficient

    # since the impulse-response of the templates is infinitely thin
    # approximate the FWHM as half the pixel width
    FWHM_tem = dL_tem/2.

    FWHM_avg_s = np.interp(x=L_tem, xp=l_avg, fp=FWHM_avg)

    if plot == True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        for z in [0.00, 0.01, 0.02]:
            # get sigma for a bunch of different redshifts
            FWHM_diff_ = np.sqrt(
                (FWHM_avg_s / (1. + z))**2. - FWHM_tem**2.)
            sigma_ = FWHM_diff_/2.355/dL_tem
            ax.plot(L_tem, sigma_,
                    label='z = {:.3f}'.format(z))
        ax.legend(loc='best')
        ax.set_xlabel(r'$\lambda[\AA]$')
        ax.set_ylabel(r'$\frac{R_{tem}}{R_{spec}}$')
        plt.tight_layout()
        plt.show()

    logL_ifu = np.log(L_ifu)

    velscale_ifu = np.asarray(
        (np.log(L_ifu[1]/L_ifu[0]) * c.c).to(u.km/u.s))

    print 'Constructing spectral library files...'

    #
    # file format is st-<REDSHIFT>.fits
    # <REDSHIFT> is of form 0.XXXX

    for i, z in enumerate([0.01]):
        print 'Template file {1} @ z = {0:.4f}'.format(z, i)

        FWHM_diff = np.sqrt(
            (FWHM_avg_s / (1. + z))**2. - FWHM_tem**2.)
        sigma = FWHM_diff/2.355/dL_tem

        a_f = convolve_variable_width(tems.data, sigma, prec=2.)

        fname2 = 'stellar_libraries/st-{0:.4f}.fits'.format(z)
        print '\tMaking HDU:', fname2

        blurred_hdu = fits.PrimaryHDU(a_f)
        blurred_hdu.header = tems.header
        blurred_hdu.header['z'] = z
        blurred_hdu.writeto(fname2, clobber=True)
