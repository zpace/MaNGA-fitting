import numpy as np
import matplotlib.pyplot as plt

import astropy.io.fits as fits
import astropy.table as table

from astropy import units as u, constants as c
import ppxf_util as util
from scipy import ndimage
import manga_tools as m


def setup_stellar_library_file(fname_tem, velscale, FWHM_gal, z):
    '''
    using a single FITS file with a T/Z/L grid of spectra,
    make a spectral library at the correct velocity scale
    '''

    SSPs_hdu = fits.open(fname_tem)
    h = SSPs_hdu[0].header

    # for some reason, the header keywords are getting messed up
    # when they're pushed to the file
    # so we need to manually get the shapes in each axis
    # (axes are more or less hard-coded in)

    logL = np.linspace(h['CRVAL3'],
                       h['CRVAL3'] + SSPs_hdu[0].data.shape[2]*h['CDELT3'],
                       SSPs_hdu[0].data.shape[2])  # base e

    L = np.exp(logL)  # also base e

    lam_range_tem = np.array([L[0], L[-1]])

    FWHM_tem = (L[1:] - L[:-1])
    FWHM_tem = np.append(FWHM_tem,
                         [(FWHM_tem[-1] - FWHM_tem[-2]) + FWHM_tem[-1]])

    FWHM_gal /= (1 + z)

    FWHM_dif = np.sqrt(FWHM_gal.value**2)  # - FWHM_tem**2)
    # Sigma difference in pixels
    sigma = (FWHM_dif/2.355/FWHM_tem)
    # print sigma
    a_f = convolve_variable_width(SSPs_hdu[0].data, sigma)

    fname2 = 'stellar_libraries/st-{0:.4f}.fits'.format(z)
    print '\tMaking HDU:', fname2

    blurred_hdu = fits.PrimaryHDU(a_f)
    blurred_hdu.header = h
    blurred_hdu.header['z'] = z
    blurred_hdu.writeto(fname2, clobber=True)
    return blurred_hdu


def convolve_variable_width(a, sig):
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
    '''

    assert (len(sig) == a.shape[-1]), '\tLast dimension of `a` must equal \
        length of `sig` (each element of a must have a convolution width)'

    sig0 = sig.max()  # the "base" width that results in minimal blurring
    n = np.rint(sig0/sig).astype(int)
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
    a_w_f = ndimage.gaussian_filter1d(a_w, sig0, axis=-1)
    # print a_w_f.shape # should be the same as the original shape

    # and downselect the pixels (take approximate middle of each slice)
    # f is a mask that will be built and applied to a_w_f

    # un-warp the newly-convolved array by selecting only the slices
    # in dimension -1 that are in nm
    a_f = a_w_f[:, :, nm]

    return a_f


def setup_MaNGA_stellar_libraries(fname_ifu, fname_tem):
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
    specres, specres_h = \
        MaNGA_hdu['SPECRES'].data, MaNGA_hdu['SPECRES'].header

    print 'Constructing wavelength grid...'
    lam_spec = np.logspace(glob_h['CRVAL3'], glob_h['CRVAL3'] +
                           glob_h['NAXIS3'] * glob_h['CD3_3'],
                           glob_h['NAXIS3'], base=10.)  # base 10

    loglam_spec = np.log(lam_spec)

    # NOW convert to base e
    velscale = np.asarray(
        (np.log(lam_spec[1]/lam_spec[0]) * c.c).to(u.km/u.s))

    FWHM_gal_pix = 3.

    FWHM_gal = FWHM_gal_pix * velscale / (u.km / u.s)

    print 'Constructing spectral library files...'

    #
    # file format is st-<REDSHIFT>.fits
    # <REDSHIFT> is of form 0.XXXX

    for z in np.arange(0., .18, .01):
        print 'z: {0:.4f}'.format(z)
        setup_stellar_library_file(fname_tem, velscale, FWHM_gal, z)
