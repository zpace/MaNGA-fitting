import numpy as np
import matplotlib.pyplot as plt

import astropy.io.fits as fits
import astropy.table as table

from astropy import units as u, constants as c
import ppxf_util as util
from ppxf import ppxf
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

        a_f = convolve_variable_width(tems.data, sigma, prec=3.)

        fname2 = 'stellar_libraries/st-{0:.4f}.fits'.format(z)
        print '\tMaking HDU:', fname2

        blurred_hdu = fits.PrimaryHDU(a_f)
        blurred_hdu.header = tems.header
        blurred_hdu.header['z'] = z
        blurred_hdu.writeto(fname2, clobber=True)


def ppxf_run_MaNGA_galaxy(ifu, fname_tem):
    plate = ifu[0]
    ifudsgn = ifu[1]
    fname_ifu = 'manga-{}-{}-LOGCUBE.fits.gz'.format(plate, ifudsgn)

    # first read in templates
    # templates have already been convolved to the proper resolution
    tems = fits.open(fname_tem)
    htems = tems[0].header
    dtems = tems[0].data

    logL_tem = np.linspace(
        htems['CRVAL3'],
        htems['CRVAL3'] + dtems.shape[2]*htems['CDELT3'],
        dtems.shape[2])  # base e

    L_tem = np.exp(logL_tem)

    dtems_med = np.median(dtems)
    dtems /= dtems_med

    # now read in IFU
    ifu = fits.open(fname_ifu)
    ifu_flux = ifu['FLUX'].data
    ifu_ivar = ifu['ivar'].data
    ifu_mask = ifu['MASK'].data
    L_ifu = ifu['WAVE'].data
    logL_ifu = np.log(L_ifu)

    # now read in drpall
    drpall = table.Table.read(m.drpall_loc + 'drpall-v1_3_3.fits',
                              format='fits')
    objconds = drpall['plateifu'] == '{}-{}'.format(plate, ifudsgn)
    obj = drpall[objconds]

    print obj['plateifu', 'nsa_redshift', 'nsa_vdisp', 'ebvgal']

    c = 299792.458
    dl = L_tem[0] - L_ifu[0]
    dv = c * (logL_tem[0] - logL_ifu[0])  # km/s
    vel = obj['nsa_redshift'] * c
    veldisp = obj['nsa_vdisp']
    if veldisp < 0.:  # deal with non-measured veldisps
        veldisp = 300.

    velscale = (logL_ifu[1] - logL_ifu[0])*c
    moments = 4
    regul_err = .004
    ebvgal = obj['ebvgal']
    dv = c*(logL_tem[0] - logL_ifu[0])

    start = [vel, veldisp]

    ifu_fits = np.nan * np.ones(ifu_flux.shape[:-1])

    spaxels = which_spaxels(fname_ifu)

    for spaxel in spaxels:
        if spaxel['good'] == True:
            gridx, gridy = spaxel['gridx'], spaxel['gridy']
            goodpixels = (ifu_mask[:, gridx, gridy] & 10)
            goodpixels *= (ifu_mask[:, gridx, gridy] & 8)

            galaxy = ifu_flux[:, gridx, gridy]
            med = np.median(galaxy)

            pp = ppxf(templates=dtems,
                      galaxy=galaxy/med,
                      noise=1./np.sqrt(ifu_ivar[:, gridx, gridy]),
                      goodpixels=goodpixels, vsyst=dv,
                      velScale=velscale, moments=moments, degree=-1,
                      mdegree=-1, regul=1./regul_err, reddening=ebvgal,
                      clean=False, lam=L_ifu)
            ifu_fits[spaxel[gridx, gridy]] = pp
            break


def which_spaxels(fname_ifu):
    '''
    list of spaxel indices in the order that they should be run
    '''

    ifu = fits.open(fname_ifu)
    ifu_flux = ifu['FLUX'].data
    r_im = ifu['RIMG'].data

    # now figure out where in the ifu the center of the galaxy is
    # and use that info to figure out where in the galaxy to start

    NL, NX, NY = ifu_flux.shape
    # print NX
    # print pixpos_x.shape, pixpos_y.shape, r_im.shape

    pos_x = np.linspace(-0.5*NX/2., 0.5*NX/2., NX)
    pos_y = np.linspace(-0.5*NY/2., 0.5*NY/2., NY)
    pixpos_x, pixpos_y = np.meshgrid(pos_x, pos_y)

    peak_inds = np.unravel_index(np.argmax(r_im), pixpos_x.shape)
    # print peak_inds

    peak_distance = np.sqrt((pixpos_x - pos_x[peak_inds[1]])**2. +
                            (pixpos_y - pos_y[peak_inds[0]])**2.)

    grid = np.indices(peak_distance.shape)
    gridx, gridy = grid[1], grid[0]

    xnew = pixpos_x - peak_inds[1]
    ynew = pixpos_y - peak_inds[0]

    theta_new = np.arctan2(ynew, xnew)

    # peak_distance and theta_new are coords relative to ctr of galaxy

    # plt.imshow(r_im, origin='lower', aspect='equal')
    # plt.scatter(peak_inds[1], peak_inds[0])
    # plt.show()

    # now make a table out of everything, so that it can get sorted

    pixels = table.Table()
    pixels['good'] = m.good_spaxels(ifu).flatten()
    pixels['gridx'] = gridx.flatten()
    pixels['gridy'] = gridy.flatten()
    pixels['r'] = peak_distance.flatten()
    pixels['theta'] = theta_new.flatten()
    pixels.sort(['r', 'theta'])
    pixels['order'] = range(len(pixels))

    print pixels

    '''plt.scatter(pixels['gridx'], pixels['gridy'],
                c=pixels['order']*pixels['good'], s=5)
    plt.colorbar()
    plt.show()'''

    return pixels
