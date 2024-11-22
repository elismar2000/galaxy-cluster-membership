#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import sigma_clip
import astropy.constants as cons

import scipy.integrate as integrate
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy import special

from sklearn.base import BaseEstimator
from sklearn.neighbors import BallTree

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.3, Om0=0.27, Tcmb0=2.725)

import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc, rcParams

f = 0.8

fontsize = 35 * f
labelsize = 35 * f

rc("xtick", labelsize=fontsize * f)
rc("ytick", labelsize=fontsize * f)
rcParams["axes.linewidth"] = 5.3 * f
rcParams["xtick.major.width"] = 5.3 * f
rcParams["xtick.minor.width"] = 5.3 * f
rcParams["ytick.major.width"] = 5.3 * f
rcParams["ytick.minor.width"] = 5.3 * f
rcParams["xtick.major.size"] = 12.5 * f
rcParams["xtick.minor.size"] = 6.5 * f
rcParams["ytick.major.size"] = 12.5 * f
rcParams["ytick.minor.size"] = 6.5 * f

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 50 * f,
        }

cm1 = plt.cm.get_cmap('jet')
cm2 = plt.cm.get_cmap('rainbow')
cm3 = plt.cm.get_cmap('gnuplot2')

color_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
colors1 = [cm1(value) for value in color_values]
colors3 = [cm3(value) for value in color_values]



#Function to compute galaxies' clustercentric distances
def clustercentric_distance(ra, dec, ra0, dec0):
    '''
    Calculate the clustercentric distance of objects given their RA and DEC
    
    
    '''
    c0 = SkyCoord(ra=ra0, dec=dec0, frame='icrs', unit='deg')
    
    sep = []
    for (i,j) in zip(ra,dec):
        c = SkyCoord(i*u.degree, j*u.degree)
        sep.append(c0.separation(c).degree) 
    sep = np.asarray(sep)
    return sep



#Define sigma_nmad function
def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))


training_set = pd.read_csv("testing_dataset_photoz.csv", usecols=['r_PStotal', 'zml', 'z'])

rmag_train = training_set["r_PStotal"]
zml_train = training_set["zml"]
z_train = training_set["z"]

fontsize=20
labelsize=15
fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(121)

n, bins, patches = ax1.hist(rmag_train, histedges_equalN(rmag_train, 25), histtype='bar', rwidth=0.8)
ax1.set_xlabel("r_PStotal", fontsize=fontsize)
ax1.set_ylabel("number of objects", fontsize=fontsize)
ax1.set_title("r_PStotal histogram with fixed number of objects per bin", fontsize=fontsize*0.6)

mag = np.array(bins)
sigma_nmad = []
mag_bins = []

for i in range(1, len(mag), 1):
    mask_i = (rmag_train > mag[i-1]) & (rmag_train < mag[i])
    delta_z_i = zml_train[mask_i] - z_train[mask_i]
    sigma_nmad.append(np.abs(1.48 * np.median(abs(delta_z_i - np.median(delta_z_i)) / (1 + z_train[mask_i]))))
    mag_bins.append("{:.3f} <= r_mag <= {:.3f}".format(mag[i-1], mag[i]))

    
def func1(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

sigma_nmad_new = np.concatenate((np.array([2e-3]), sigma_nmad))
mag_new = np.concatenate((np.array([13.00]), mag))

xdata = mag_new[:-1]
ydata = sigma_nmad_new

popt1, pcov1 = curve_fit(func1, xdata, ydata)
print(popt1)

ax2 = fig.add_subplot(122)

xnew = np.linspace(xdata[0], xdata[-1], 100)
# ax.scatter(xdata, ydata)
ax2.bar(mag_new[:-1], sigma_nmad_new, alpha=0.8, width=0.2)
ax2.plot(xnew, func1(xnew, *popt1), linewidth=2.0, color='red', label="Fitting function (3rd order polynomial)")
ax2.set_xlabel("Magnitude (r_PStotal)", fontsize=fontsize)
ax2.set_ylabel(r"Mean $\sigma_{NMAD}$", fontsize=fontsize)
ax2.xaxis.set_tick_params(labelsize=labelsize, width=3, rotation=90)
ax2.yaxis.set_tick_params(labelsize=labelsize, width=3)
ax2.legend()

fig.tight_layout(pad=1.0)
    
def sigma_nmad_(r):
    if (r > mag_new[:-1][2]): sigma_nmad = func1(r, *popt1)
    elif (r <= mag_new[:-1][2]): sigma_nmad = func1(mag_new[:-1][2], *popt1)
    return sigma_nmad



#Function to define photo-z PDFs
def calc_PDF_series(weights, means, stds, x_range=None, optimize_zml=False):
    '''
    Returns a list of PDFs calculated as a combination of Gaussian functions

    Keyword arguments:
    x            -- Photometric redshift range for which the PDF should be calculated
    weights      -- Weight of the Gaussian components
    means        -- means of the Gaussian components
    stds         -- Standard deviation of the Gaussian components
    optimize_zml -- If the single-point estimate of photometric redshift should be optimized (if True, it will be
                    determined on a finer grid of points)
    '''
    
    if x_range is None:
        x = np.arange(-0.005, 1+0.001, 0.001) 
    else:
        x = x_range
                      
    # Convert columns from string to lists
    if type(weights) != np.ndarray:
        weights = np.array(weights)
        means   = np.array(means)
        stds    = np.array(stds)

    # Calculating PDFs and optimizing photo-zs (optional)
    PDFs           = []
    optimized_zmls = np.empty(len(means))
    
    if np.ndim(weights) == 2: # weights, means, and stds are 2D arrays 
        for i in range(len(weights)):
            PDF = np.sum(weights[i]*(1/(stds[i]*np.sqrt(2*np.pi))) * np.exp((-1/2) * ((x[:,None]-means[i])**2)/(stds[i])**2), axis=1)
            PDF = PDF/np.trapz(PDF, x)
            PDFs.append(PDF)
        zmls = x[np.argmax(PDFs, axis=1)]
        
    if np.ndim(weights) == 1:
        PDF  = np.sum(weights*(1/(stds*np.sqrt(2*np.pi))) * np.exp((-1/2) * ((x[:,None]-means)**2)/(stds)**2), axis=1)
        PDFs = PDF/np.trapz(PDF, x)
        zmls = x[np.argmax(PDFs)]

    if optimize_zml == True:
        for i in range(len(weights)):
            # First run
            optimized_x   = np.linspace(zmls[i]-0.002, zmls[i]+0.002, 500, endpoint=True)
            optimized_PDF = np.sum(weights[i]*(1/(stds[i]*np.sqrt(2*np.pi))) * np.exp((-1/2) * ((optimized_x[:,None]-means[i])**2)/(stds[i])**2), axis=1)
            optimized_zml = optimized_x[np.argmax(optimized_PDF)]

            # Second run
            optimized_x   = np.linspace(optimized_zml-0.001, optimized_zml+0.001, 300, endpoint=True)
            optimized_PDF = np.sum(weights[i]*(1/(stds[i]*np.sqrt(2*np.pi))) * np.exp((-1/2) * ((optimized_x[:,None]-means[i])**2)/(stds[i])**2), axis=1)
            optimized_zmls[i] = optimized_x[np.argmax(optimized_PDF)]

        zmls = optimized_zmls
                
    return PDFs, zmls, x



#Function to compute P_pz_G and P_pz_F, two entries of the membership probability equation
def zp_mem_prob_george11(zps, zc, means, weights, stds, sigma):
    '''
    Compute cluster and field membership probabilities given the i-th galaxy photo-z PDF parameters

    Parameters
    ----------
    i : integer
        iteration index for the i-th galaxy

    zc : float
        cluster redshift

    means : array
        means of the photo-z PDFs

    weights : array
        weights of the photo-z PDFs

    stds : array
        standard deviations of the photo-z PDFs

    sigma : float
        width of the gaussian representing the cluster in photo-z space
    '''

    #This gaussian represents the cluster
    def gaussian(x, zc, sigma):
        y = lambda x, zc, sigma: (1 / (sigma * np.sqrt(2* np.pi))) * np.exp(-(x - zc)**2 / (2*sigma**2)) 
        a = 1 / integrate.quad(y, 0.0, np.inf, args=(zc, sigma))[0]
        return a * y(x, zc, sigma)


    P_pz_C_i_array = np.zeros(len(zps))
    P_pz_F_i_array = np.zeros(len(zps))
    for i in range(len(zps)): 
        means_ = means[i]
        weights_ = weights[i]
        stds_ = stds[i]
        
        pdfs, zmls, x = calc_PDF_series(weights_, means_, stds_) 
        
        pdfs_interp = interp1d(x, pdfs)
        a = 1 / integrate.quad(pdfs_interp, 0.0, 1.0)[0]

        pdf_func = lambda x: a * pdfs_interp(x)
            
        if zc - 3*sigma <= 0:
            P_pz_C = integrate.quad(lambda x, sigma, zc: pdf_func(x) * gaussian(x, zc, sigma), 0.0, zc + 3*sigma, args=(sigma, zc))[0] 
            P_pz_F = integrate.quad(lambda x, sigma: pdf_func(x) / (6*sigma), 0.0, zc + 3*sigma, args=(sigma))[0]
            
        if zc - 3*sigma > 0:
            P_pz_C = integrate.quad(lambda x, sigma, zc: pdf_func(x) * gaussian(x, zc, sigma), zc - 3*sigma, zc + 3*sigma, args=(sigma, zc))[0] 
            P_pz_F = integrate.quad(lambda x, sigma: pdf_func(x) / (6*sigma), zc - 3*sigma, zc + 3*sigma, args=(sigma))[0]
        
        P_pz_C_i_array[i] = P_pz_C
        P_pz_F_i_array[i] = P_pz_F

    return P_pz_C_i_array, P_pz_F_i_array



def P_pz_v2(zc, rmag, means, weights, stds, fz=3.0):
    
    '''
    Compute the membership probability of galaxies in a cluster by integrating the photo-z PDF
    in the interval 0 < zc - fz*sigma_i*(1+zc) < zc + fz*sigma_i*(1+zc)
    
    zc : float
        redshift of the cluster
    
    rmag : array
        r-band magnitude of the galaxies in which the membership will be applied
    
    means : array
        means of the photo-z PDFs

    weights : array
        weights of the photo-z PDFs

    stds : array
        standard deviations of the photo-z PDFs
    
    fz : float
        defines the range of photo-zs for candidate cluster members in terms of sigma_nmad
    '''
    
    sigma = np.array([sigma_nmad_(r) for r in rmag])
    
    
    P_pz_C_array = np.zeros(len(rmag))
    for i in range(len(rmag)):

        sigma_i = sigma[i]
       
        means_ = means[i]
        weights_ = weights[i]
        stds_ = stds[i]    

        pdfs, zmls, x = calc_PDF_series(weights_, means_, stds_)    
        pdfs_interp = interp1d(x, pdfs)
        a = 1 / integrate.quad(pdfs_interp, 0.0, 1.0)[0]
        pdf_func = lambda x: a * pdfs_interp(x)

        if zc - fz*sigma_i*(1+zc) <= 0:
            P_pz_C = integrate.quad(lambda x: pdf_func(x), 0.0, zc + fz*sigma_i*(1+zc))[0] 

        if zc - fz*sigma_i*(1+zc) > 0:
            P_pz_C = integrate.quad(lambda x: pdf_func(x), zc - fz*sigma_i*(1+zc), zc + fz*sigma_i*(1+zc))[0] 
                                    
        P_pz_C_array[i] = P_pz_C
        
    return P_pz_C_array



def P_pz_v2_mock(zc, zps, rmag, fz=3.0):
    
    '''
    Perform the membership of galaxies in a cluster mock
    
    zc : redshift of the cluster
    
    zps : photometric redshift of the galaxies in which the membership will be applied
    
    rmag : r-band magnitude of the galaxies in which the membership will be applied
    
    fz : defines the range of photo-zs for candidate cluster members in terms of sigma_nmad
    '''
    
    sigma = np.array([sigma_nmad(r) for r in rmag])
    
    #We will simulate a gaussian PDF for the photo-zs of galaxies in the mock
    def pdf(x, z, sigma):
        y = lambda x, z, sigma: (1 / (sigma * np.sqrt(2* np.pi))) * np.exp(-(x - z)**2 / (2*sigma**2)) 
        a = 1 / integrate.quad(y, 0.0, np.inf, args=(z, sigma))[0]
        return a * y(x, z, sigma)
    
    
    P_pz_C_array = np.zeros(len(zps))
    for i in range(len(zps)):
        z = zps[i]
        sigma_i = sigma[i]

        if zc - fz*sigma_i*(1+zc) <= 0:
            P_pz_C = integrate.quad(lambda x, z, sigma_i: pdf(x, z, sigma_i), 0.0, zc + fz*sigma_i*(1+zc), args=(z, sigma_i))[0] 

        if zc - fz*sigma_i*(1+zc) > 0:
            P_pz_C = integrate.quad(lambda x, z, sigma_i: pdf(x, z, sigma_i), zc - fz*sigma_i*(1+zc), zc + fz*sigma_i*(1+zc), args=(z, sigma_i))[0] 
                                    
        P_pz_C_array[i] = P_pz_C
        
    return P_pz_C_array


#Aqui ainda teria que implementar direito o perfil de NFW, mas como eu não tô utilizando ele por enquanto
#achei conveniente deixar por isso mesmo
#Ou seja, por favor não usar o perfil de NFW pois aqui está apenas o perfil de densidade, e não o cumulativo
def radial_mem_prob(clc_dist, rc, name, fz, cluster_profile="NFW", plot=True):
    '''
    clc_dist : clustercentric distances of the galaxies
    
    rc : cluster radius
    
    cluster_profile : "NFW" or "power-law"
    
    plot : bool, to plot or not
    '''
    
    # creating CDF of projected clustercentric distances
    Hz, Rcz = np.histogram(clc_dist, bins=100, density=True)
    dx = Rcz[1] - Rcz[0]
    Fz = np.cumsum(Hz)*dx

    
    #Expression for the cumulative density of the field
    rho_f = lambda R, w2: np.pi * w2 * R**2
    
    
    #Expression for the cumulative density of the cluster (can be either a power-law of a NFW profile)
    if cluster_profile == "power-law":       
        rho_c = lambda R, w1, alpha: 2 * np.pi * w1 * (R**(2 - alpha))/(2 - alpha)
        rho = lambda R, w1, w2, alpha: rho_c(R, w1, alpha) + rho_f(R, w2)
        
        params = curve_fit(rho, Rcz[1:], Fz, bounds=((0, 0, 0), (np.inf, np.inf, 3)))[0]
        w1, w2, alpha = params

    
    elif cluster_profile == "NFW":
        def NFW_profile(R, rho_s, r_s):   
            R = np.deg2rad(R)
            Sig_1h = np.zeros_like(R)
            for r, i in zip(R, range(len(R))):
                x = r/r_s
                if 1 - np.around(x, 8) == 0.0:
                    F = 1/3
                else:
                    if x < 1:
                        C = np.arccosh(1/x)
                    elif x > 1:
                        C = np.arccos(1/x)
                    F = 1/(x*2 - 1) * ( 1 - C/np.sqrt(np.abs(x*2 - 1)) )    
                Sig_1h[i] = 2*r_s*rho_s * F       
            return Sig_1h
    
        rho = lambda R, w2, rho_s, r_s: NFW_profile(R, rho_s, r_s) + rho_f(R, w2)
        
        params = curve_fit(rho, Rcz[1:], Fz)[0]
        w2, rho_s, r_s = params
   

    else: 
        print("'cluster_profile' parameter should be either 'NFW' or 'power-law'")
    
    
    rho_fitted = rho(Rcz[1:], *params)
    
    
    if plot == True:
        if cluster_profile == "power-law":
            fig = plt.figure(figsize=(10, 10))

            ax = fig.add_subplot(111)    
            ax.plot(Rcz[1:]/rc, Fz, lw=5.0, c=colors3[1], label="Projected radial distribution of galaxies", alpha=0.3)
            ax.plot(Rcz[1:]/rc, rho_fitted, lw=4.0, ls="--", label="Best combined fit", c=colors3[5])
            ax.plot(Rcz[1:]/rc, rho_c(Rcz[1:], w1, alpha), lw=3.0, ls="-", label="Best cluster fit", c=colors3[3], alpha=0.7)
            ax.plot(Rcz[1:]/rc, rho_f(Rcz[1:], w2), lw=3.0, ls="-", label="Best field fit", c=colors3[6], alpha=0.7)
            
            ax.text(0.0, 0.8, s=r"$w_1 = {:.2f}$".format(w1), fontsize=15)
            ax.text(0.0, 0.75, s=r"$\alpha = {:.2f}$".format(alpha), fontsize=15)
            ax.text(0.0, 0.7, s=r"$w_2 = {:.2f}$".format(w2), fontsize=15)

            ax.set_xlabel(r"$R_c/r_{200}$", fontdict=font)
            ax.set_ylabel("CDF", fontdict=font)
            ax.legend(fontsize=15)

            #k is a normalizing factor for the probabilities
            k = rho_c(clc_dist, w1, alpha) + rho_f(clc_dist, w2)
            Pmem_R_C = rho_c(clc_dist, w1, alpha) / k
            # plt.savefig("../figures/membership-testing-on-mocks/power-law-radial-fit_fz{}_mock{}.png".format(fz, name), dpi='figure', format='png')

        elif cluster_profile == "NFW":
            fig = plt.figure(figsize=(10, 10))

            ax = fig.add_subplot(111)    
            ax.plot(Rcz[1:]/rc, Fz, lw=5.0, c=colors3[1], label="Projected radial distribution of galaxies", alpha=0.3)
            ax.plot(Rcz[1:]/rc, rho_fitted, lw=4.0, ls="--", label="Best combined fit", c=colors3[5])
            ax.plot(Rcz[1:]/rc, NFW_profile(Rcz[1:], rho_s, r_s), lw=3.0, ls="-", label="Best cluster fit", c=colors3[3], alpha=0.7)
            ax.plot(Rcz[1:]/rc, rho_f(Rcz[1:], w2), lw=3.0, ls="-", label="Best field fit", c=colors3[6], alpha=0.7)

            ax.set_xlabel(r"$R_c/r_{200}$", fontdict=font)
            ax.set_ylabel("CDF", fontdict=font)
            ax.legend(fontsize=15)
            # plt.savefig("../figures/membership-testing-on-msocks/NFW-radial-fit_fz{}_mock{}.png".format(fz, name), dpi='figure', format='png')

            #k is a normalizing factor for the probabilities
            k = NFW_profile(clc_dist, rho_s, r_s) + rho_f(clc_dist, w2)
            Pmem_R_C = NFW_profile(clc_dist, rho_s, r_s) / k
            Pmem_R_F = rho_f(clc_dist, w2) / k
   

    Pmem_R_F = rho_f(clc_dist, w2) / k
    
    return Pmem_R_C, Pmem_R_F, w1, w2, alpha


#Function to perform sigma_clipping
def sigma_clipping(zs, ids, zlower, zupper, sigma):
    '''
    Perform a cluster spectroscopic membership using sigma clipping
    '''
    # It is necessary to restrict a little the sample.  We have to look at this cluster by cluster
    cluster_max_cut = (zs > zlower) & (zs < zupper)

    cluster_sig = sigma_clip(zs[cluster_max_cut], sigma=sigma, cenfunc='median', stdfunc='mad_std') 

    # Objects selected after applying the 3sigmaclipping cut
    specz_members = zs[cluster_max_cut][~cluster_sig.mask]
    id_members = ids[cluster_max_cut][~cluster_sig.mask]
    
    return specz_members, id_members



def spec_members_vesc(m200, rc, zc, zs, f):
    '''
    m200 : virial mass of the cluster in units of 10^14 h^-1 Msun
    rc : virial radius of the cluster in units of h^-1 Mpc
    zc : cluster redshift
    zs : spectroscopic redshift of the galaxies
    f : multiplying factor for v_esc
    '''
    
    #escape velocity of the cluster in km s^-1, acoording to Lima-Dias+21
    v_esc = 927 * np.sqrt((m200 / (1e14 * u.Msun / cosmo.h))) * np.sqrt((u.Mpc / cosmo.h) / rc)
    
    #peculiar velocity of the cluster
    v_pec = cons.c.to(u.km / u.s) * (zs - zc) / (1 + zc)
    
    #selecting member and infalling galaxies
    mask_members = np.abs(v_pec.value) < f * v_esc
    
    return mask_members



#Function to compute priors in the same way as in George+11
def priors_george11(rmags, zps, clc_dist, zc, rc, plot=False):
    
    '''
    Method to calculate priors as a function of redshift and magnitude bins,
    in a very similar way as defined in the original paper of the membership method (George+11).
    It seems to work only for clusters at higher redshifts (z >~ 0.5), otherwise it underestimates the probability
    of galaxies belonging to the field.

    rmags : array containing the r magnitude of galaxies in the cluster
    
    zps : array containing the photo-zs of galaxies in the cluster
    
    clc_sep : array containing the clustercentric distances of the objects
    
    zc : redshift of the cluster
    
    R200 : virial radius of the cluster in deg

    plot : True of False
        define whether to plot a figure ilustrating the computation of the priors or not
    '''
    
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(111)
    color = cm.rainbow(np.linspace(0, 1, 14))
    
    dr = 0.9 #width of magnitude bins
    r_min = 12.9 #minimum r magnitude
    r_max = 22.0 #maximum r magnitude
    dz = 0.05 #width of photo-z bins
    z_max = 0.7 #maximum range of photo-zs
    
    priors = {}
    
    for r, i in zip(np.arange(r_min, r_max, dr), range(len(np.arange(r_min, r_max, dr)))): 
        mask_r = (rmags > r) & (rmags < r+dr)
        sigma = sigma_nmad(np.mean(rmags[mask_r]))
        
        mask_field = ~((clc_dist < 3*rc) & (zps > zc - 5*sigma) & (zps < zc + 5*sigma))
        #I only want galaxies in this particular magnitude bin and those belonging to the field
        mask = mask_field & mask_r
        
        Nf = np.histogram(zps[mask], range=(0, z_max), bins=int(np.round(z_max/dz)))[0]
        zp_bins = np.histogram(zps[mask], range=(0, z_max), bins=int(np.round(z_max/dz)))[1][:-1] + dz/2

        #The field angular area will be different when we consider the photo-z region of the cluster and the photo-z region beyond the cluster
        nf = np.concatenate((Nf[(zp_bins < zc + 5*sigma) & (zp_bins > zc - 5*sigma)] / (dz * np.pi * ((5*rc)**2 - (3*rc)**2)), 
                             Nf[(zp_bins > zc + 5*sigma) | (zp_bins < zc - 5*sigma)] / (dz * np.pi * (5*rc)**2)))
        
        #Interpolate the points and get the value of nf at zc
        nf_function = interp1d(x=zp_bins, y=nf, kind="linear")
        zp_bins_interp = np.linspace(zp_bins.min(), zp_bins.max(), 100)
        nf_interp = nf_function(zp_bins_interp)
        
        nf_cluster = nf_function(zc)
        
        #Computing volume of the cluster (considering it a cylinder) and the number of field galaxies in this volume
        if (zc - 5*sigma <= 0): volume_cluster = np.pi * (zc + 3*sigma) * rc**2
        if (zc - 5*sigma > 0): volume_cluster = np.pi * 6*sigma * rc**2
            
        Nf_cluster = nf_cluster * volume_cluster
            
        #Total number of galaxies in the cluster volume
        Ntot = np.sum((zps > zc - 3*sigma) & (zps < zc + 3*sigma) & (clc_dist < rc))
        
        #Computing the prior P(g in F)
        P_ginF = Nf_cluster / Ntot
        
        priors["{:.2f} < r < {:.2f}".format(r, r+dr)] = P_ginF
  
        if plot == True:
            ax1.scatter(zp_bins, nf, color=color[i], lw=5, marker="s", s=20)
            ax1.plot(zp_bins_interp, nf_interp, color=color[i], lw=3, label=r"${:.2f} < r < {:.2f}$".format(r, r+dr))


    if plot == True:
        #ax1.set_title(cl_names[cluster], fontdict=font)

        ax1.set_xlabel(r"$z_p$", fontdict=font)
        ax1.set_ylabel(r"$N_f / dz / d \Omega$", fontdict=font)

        ax1.axvline(zc, color="black", label=r"$z_C$")
        ax1.axvline(zc + 5*sigma, color="blue", label=r"$z_C + 5\sigma$")

        ax1.legend(fontsize=10)
        
    return priors
    


#AstroML.KNeighborsDensity
def n_volume(r, n):
    """compute the n-volume of a sphere of radius r in n dimensions"""
    return np.pi ** (0.5 * n) / special.gamma(0.5 * n + 1) * (r ** n)


class KNeighborsDensity(BaseEstimator):
    """K-neighbors density estimation
    Parameters
    ----------
    method : string
        method to use.  Must be one of ['simple'|'bayesian'] (see below)
    n_neighbors : int
        number of neighbors to use
    Notes
    -----
    The two methods are as follows:
    - simple:
        The density at a point x is estimated by n(x) ~ k / r_k^n
    - bayesian:
        The density at a point x is estimated by n(x) ~ sum_{i=1}^k[1 / r_i^n].
    See Also
    --------
    KDE : kernel density estimation
    """
    def __init__(self, method='bayesian', n_neighbors=10):
        if method not in ['simple', 'bayesian']:
            raise ValueError("method = %s not recognized" % method)

        self.n_neighbors = n_neighbors
        self.method = method

    def fit(self, X):
        """Train the K-neighbors density estimator
        Parameters
        ----------
        X : array_like
            array of points to use to train the KDE.  Shape is
            (n_points, n_dim)
        """
        self.X_ = np.atleast_2d(X)

        if self.X_.ndim != 2:
            raise ValueError('X must be two-dimensional')

        self.bt_ = BallTree(self.X_, metric="haversine")

        return self

    def eval(self, X):
        """Evaluate the kernel density estimation
        Parameters
        ----------
        X : array_like
            array of points at which to evaluate the KDE.  Shape is
            (n_points, n_dim), where n_dim matches the dimension of
            the training points.
        Returns
        -------
        dens : ndarray
            array of shape (n_points,) giving the density at each point.
            The density will be normalized for metric='gaussian' or
            metric='tophat', and will be unnormalized otherwise.
        """
        X = np.atleast_2d(X)
        if X.ndim != 2:
            raise ValueError('X must be two-dimensional')

        if X.shape[1] != self.X_.shape[1]:
            raise ValueError('dimensions of X do not match training dimension')

        dist, ind = self.bt_.query(X, self.n_neighbors, return_distance=True)

        k = float(self.n_neighbors)
        ndim = X.shape[1]

        if self.method == 'simple':
            return k / n_volume(dist[:, -1], ndim)

        elif self.method == 'bayesian':
            # XXX this may be wrong in more than 1 dimension!
            return (k * (k + 1) * 0.5 / n_volume(1, ndim)
                    / (dist ** ndim).sum(1))
        else:
            raise ValueError("Unrecognized method '%s'" % self.method)



def knn2D(x, y, K, xbins=200j, ybins=200j, **kwargs): 
    """Build 2D Nearest-Neighbor density estimate (KNN)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins, y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    knd = KNeighborsDensity(method="bayesian", n_neighbors=K)
    knd.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = knd.eval(xy_sample)
    return xx, yy, np.reshape(z, xx.shape)