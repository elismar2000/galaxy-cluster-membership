print("DEFINING SOME QUANTITIES")
#Define the volume of the cluster
#fz1 determines the extent of the "mask_cluster", fz2 determines the extent to compute P(C | zp), and fr defines the radial extent of the objects
fz1 = 3.0
fz2 = 3.0
fr = 5.0

print("LOADING PACKAGES")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.io import fits
from astropy.table import Table, join
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.constants as cons
from astropy.visualization.wcsaxes import SphericalCircle

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.3, Om0=0.27, Tcmb0=2.725)

#Another way to do this would be to add an "__init__.py" to the folder '/work/Doc/galaxy-cluster-membership/' and import it as galaxy-cluster-membership.galaxy_cluster_membership I think
#See https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
# import sys
# sys.path.insert(1, '/work/Doc/galaxy-cluster-membership/')
from galaxy_cluster_membership import *

import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc, rcParams
f = 0.8

fontsize = 35 * f
labelsize = 20 * f

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
colors2 = [cm2(value) for value in color_values]
colors3 = [cm3(value) for value in color_values]


print("LOADING MOCK")
hdul = fits.open("cluster-mocks/lightcone_zphot_subhalos_AllFrom1.fits")
scat = hdul[1].data
table = Table(scat)

magr = scat["mag_r"]
mask_mag = magr < 21

haloId = scat["haloId"][mask_mag]
galaxyId = scat["galaxyId"][mask_mag]
z_phot_original = scat["z_phot"][mask_mag]
z_spec = scat["z_app"][mask_mag]
ra = scat["RA"][mask_mag]
dec = scat["DEC"][mask_mag]
magr = scat["mag_r"][mask_mag]
magu = scat["mag_u"][mask_mag]
magg = scat["mag_g"][mask_mag]
magi = scat["mag_i"][mask_mag]
magz = scat["mag_z"][mask_mag]
firstHaloinFoFGroupId = scat["firstHaloinFoFGroupId"][mask_mag]


print("ASSOCIATING PHOTO-Z PDF PROPERTIES TO MOCK GALAXIES")
training_set_pdfs = pd.read_csv("testing_dataset_photoz_pdfs.csv", 
                   usecols=['z_x', 'u_PStotal', 'g_PStotal', 'r_PStotal', 'i_PStotal', 'z_PStotal', 
                           'pdf_weights_0', 'pdf_weights_1', 'pdf_weights_2', 'pdf_means_0', 'pdf_means_1', 
                           'pdf_means_2', 'pdf_stds_0', 'pdf_stds_1', 'pdf_stds_2', 'zml']) 

z_pdfs = training_set_pdfs['z_x']
zp_pdfs = training_set_pdfs['zml']
u_PStotal_pdfs = training_set_pdfs['u_PStotal']
g_PStotal_pdfs = training_set_pdfs['g_PStotal']
r_PStotal_pdfs = training_set_pdfs['r_PStotal']
i_PStotal_pdfs = training_set_pdfs['i_PStotal']
z_PStotal_pdfs = training_set_pdfs['z_PStotal']
weights_0_pdfs = training_set_pdfs['pdf_weights_0']
weights_1_pdfs = training_set_pdfs['pdf_weights_1']
weights_2_pdfs = training_set_pdfs['pdf_weights_2']
means_0_pdfs = training_set_pdfs['pdf_means_0']
means_1_pdfs = training_set_pdfs['pdf_means_1']
means_2_pdfs = training_set_pdfs['pdf_means_2']
stds_0_pdfs = training_set_pdfs['pdf_stds_0']
stds_1_pdfs = training_set_pdfs['pdf_stds_1']
stds_2_pdfs = training_set_pdfs['pdf_stds_2']

mock_attributes = np.array([z_spec])
obs_attributes = np.array([z_pdfs])
nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(obs_attributes.T)
d, indices = nbrs.kneighbors(mock_attributes.T)

z_phot_pdfs = np.array([zp_pdfs[i] for i in indices[:, 0]])
weights_0_pdfs_mock = np.array([weights_0_pdfs[i] for i in indices[:, 0]])
weights_1_pdfs_mock = np.array([weights_1_pdfs[i] for i in indices[:, 0]])
weights_2_pdfs_mock = np.array([weights_2_pdfs[i] for i in indices[:, 0]])
means_0_pdfs_mock = np.array([means_0_pdfs[i] for i in indices[:, 0]])
means_1_pdfs_mock = np.array([means_1_pdfs[i] for i in indices[:, 0]])
means_2_pdfs_mock = np.array([means_2_pdfs[i] for i in indices[:, 0]])
stds_0_pdfs_mock = np.array([stds_0_pdfs[i] for i in indices[:, 0]])
stds_1_pdfs_mock = np.array([stds_1_pdfs[i] for i in indices[:, 0]])
stds_2_pdfs_mock = np.array([stds_2_pdfs[i] for i in indices[:, 0]])

weights_pdfs_mock = np.vstack((weights_0_pdfs_mock, weights_1_pdfs_mock, weights_2_pdfs_mock))
means_pdfs_mock = np.vstack((means_0_pdfs_mock, means_1_pdfs_mock, means_2_pdfs_mock))
stds_pdfs_mock = np.vstack((stds_0_pdfs_mock, stds_1_pdfs_mock, stds_2_pdfs_mock))


#I have to apply the mask "firstHaloinFoFGroupId > 0" because otherwise I will have the value -9223372036854775808
#in it, which is actually a value assumed by python for the galaxies with no firstHaloinFoFGroupId assigned to it
print("COMPUTING WHAT ARE THE MOST POPULATED MOCK CLUSTERS")
halo_members_list = []
for i in range(len(np.unique(firstHaloinFoFGroupId[firstHaloinFoFGroupId > 0]))):
    halo_members = np.sum(firstHaloinFoFGroupId == np.unique(firstHaloinFoFGroupId[firstHaloinFoFGroupId > 0])[i])
    halo_members_list.append(halo_members)

halo_ids_list = np.array([])
for i_mock in np.arange(0, 500, 1):
    halo_ids = np.unique(firstHaloinFoFGroupId[firstHaloinFoFGroupId > 0])[halo_members_list == np.flip(np.sort(halo_members_list))[i_mock]]
    halo_ids_list = np.concatenate((halo_ids_list, halo_ids))

halo_ids_list_unique = []
for h in halo_ids_list:
    if ~np.isin(h, halo_ids_list_unique): halo_ids_list_unique.append(h)


i_mock_list = []
ra0_list = []
dec0_list = []
z_cluster_list = []
m200_list = []
r200_mpc_list = []
r200_deg_list = []
zp_bias_list = []
cut_zp_list = []
c_zp_max_list = []
p_zp_max_list = []
f1_zp_max_list = []
c_zp_list = []
p_zp_list = []

print("START OF THE LOOP")
for i_mock in np.arange(0, 500, 1):
    print("COMPUTING FOR CLUSTER {}".format(i_mock))
    halo_id = halo_ids_list_unique[i_mock]
    halo_mask = firstHaloinFoFGroupId == halo_id

    halo_ra = ra[halo_mask]
    halo_dec = dec[halo_mask]
    halo_zphot = z_phot_pdfs[halo_mask]
    halo_zspec = z_spec[halo_mask]
    halo_magr = magr[halo_mask]

    #Define some quantities for the cluster (center cordinates, redshift, radius, photo-z error (sigma))
    ra0 = np.mean(halo_ra)
    dec0 = np.mean(halo_dec)

    z_bins = 20
    bw = (halo_zspec.max() - halo_zspec.min()) / z_bins

    kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(halo_zspec[:, np.newaxis])
    zz = np.linspace(halo_zspec.min(), halo_zspec.max(), len(halo_zspec))[:, np.newaxis]
    log_dens_zz = kde.score_samples(zz)

    z_cluster = zz[np.argmax(np.exp(log_dens_zz))]


    fig = plt.figure(figsize=(20, 10))

    ax1 = fig.add_subplot(121)
    ax1.hist(halo_zspec, bins=z_bins, color=colors1[5], histtype='stepfilled', edgecolor='k', 
            range=[halo_zspec.min(), halo_zspec.max()])[2]

    ax1.set_title("Mock cluster {:d}".format(i_mock), fontdict=font)
    ax1.set_xlabel("z_spec", fontdict=font)
    ax1.set_ylabel("Number of objects", fontdict=font)
    ax1.tick_params(axis='x', rotation=45)

    ax2 = fig.add_subplot(122)
    ax2.plot(zz, np.exp(log_dens_zz), lw=4, color=colors1[8], label="KDE estimation")
    ax2.axvline(z_cluster, color="blue", lw=3, label="peak of redshift distribution")
    ax2.set_xlabel("z_spec", fontdict=font)
    ax2.set_ylabel("Density of objects", fontdict=font)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(fontsize=labelsize)

    fig.tight_layout(pad=3.0)
    plt.savefig("../figures/membership-testing-on-mocks/z_distribution_mock{}.png".format(i_mock), dpi='figure', format='png')


    m200 = scat["m_crit200"][halo_mask].max() * 1e10 * u.Msun / cosmo.h
    r200 = (m200 * cons.G / (100 * cosmo.H(z_cluster)**2))**(1/3)
    r200_mpc = r200.to(u.Mpc)
    kpc_scale = cosmo.kpc_proper_per_arcmin(z=z_cluster).value/60.
    r200_deg = r200_mpc.value*1000 / (kpc_scale * 3600)

    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(121)
    sc = ax1.scatter(halo_ra, halo_dec, c=halo_zphot, cmap=cm1)
    ax1.scatter(ra0, dec0, marker="X", s=100, color="red", label="Cluster center")

    ax1.set_title("Mock cluster {:d}".format(i_mock), fontdict=font)
    ax1.set_xlabel("R.A.", fontdict=font)
    ax1.set_ylabel("Dec.", fontdict=font)
    ax1.tick_params(axis='x', rotation=45)

    ax1.text(ra0+3*r200_deg, dec0+3*r200_deg, s=r"$RA_0 = {:.2f}$".format(ra0), fontsize=15)
    ax1.text(ra0+3*r200_deg, dec0+2.8*r200_deg, s=r"$Dec_0 = {:.2f}$".format(dec0), fontsize=15)
    ax1.text(ra0+3*r200_deg, dec0+2.6*r200_deg, s=r"$z_C = {:.2f}$".format(z_cluster[0]), fontsize=15)
    ax1.text(ra0+3*r200_deg, dec0+2.4*r200_deg, s=r"$m200 = {:.2f} (1e15 Msun/h)$".format(m200.value/1e15), fontsize=15)
    ax1.text(ra0+3*r200_deg, dec0+2.2*r200_deg, s=r"$r200 = {:.2f}º = {:.2f} Mpc$".format(r200_deg[0], r200_mpc.value[0]), fontsize=15)
    # ax1.text(ra0-3*r200_deg, dec0+2.0*r200_deg, s=r"$r200 = {:.2f} Mpc$".format(r200_mpc.value[0]), fontsize=15)

    c0 = SkyCoord(ra=ra0, dec=dec0, frame='icrs', unit='deg')

    r1 = SphericalCircle(c0, r200_deg*u.deg, resolution=100, vertex_unit='deg', 
                lw=3, label="Cluster radius", edgecolor="red", facecolor="None")
    ax1.add_patch(r1)

    r2 = SphericalCircle(c0, 3*r200_deg*u.deg, resolution=100, vertex_unit='deg', 
                lw=3, label=r"$3 \times$ Cluster radius", edgecolor="red", facecolor="None", ls='--')
    ax1.add_patch(r2)

    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(sc, cax=cax1, orientation='vertical')
    cbar.set_label(label="z_phot", fontdict=font)
    cbar.ax.tick_params(labelsize=labelsize)

    ax1.invert_xaxis()

    ax1.legend(fontsize=15)


    ax3 = fig.add_subplot(122)
    ax3.hist(halo_zphot, bins=z_bins, color=colors1[5], histtype='stepfilled', edgecolor='k')[2]
    ax3.set_xlabel(r"$z_p$", fontdict=font)
    ax3.set_ylabel("Number of objects", fontdict=font)
    # ax3.axvline(mean_z, color="red", lw=3, label="mean redshift")
    # ax3.axvline(median_z, color="brown", lw=3, label="median redshift")
    ax3.axvline(z_cluster, color="red", lw=3, label="peak of redshift distribution")
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(fontsize=15)
        
    fig.tight_layout(pad=3.0)

    plt.savefig("../figures/membership-testing-on-mocks/RA_DEC_z_zphot(KNN-with-z)_distribution_mock{}.png".format(i_mock), dpi='figure', format='png')


    pre_mask = (ra < ra0 + 6*r200_deg) & (ra > ra0 - 6*r200_deg) & (dec < dec0 + 6*r200_deg) & (dec > dec0 - 6*r200_deg)
    halo_mask_restricted = halo_mask[pre_mask]

    dist = clustercentric_distance(ra[pre_mask], dec[pre_mask], ra0, dec0)
    #Computing the distance between all galaxies that I know are not in the cluster and the center of the cluster
    # dist_field = clustercentric_distance(ra[~halo_mask], dec[~halo_mask], ra0, dec0)
    dist_field = dist[~halo_mask_restricted]


    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(121)
    max_dist = 5*r200_deg
    sc = ax1.scatter(ra[pre_mask][~halo_mask_restricted][dist_field < max_dist], dec[pre_mask][~halo_mask_restricted][dist_field < max_dist], 
                    c=dist_field[dist_field < max_dist], cmap=cm.Set3, s=10, label="Field")

    sc2 = ax1.scatter(halo_ra, halo_dec, c="blue", label="Cluster")

    c0 = SkyCoord(ra=ra0, dec=dec0, frame='icrs', unit='deg')

    r1 = SphericalCircle(c0, r200_deg*u.deg, resolution=100, vertex_unit='deg', 
                lw=3, label="Cluster radius", edgecolor="red", facecolor="None")
    ax1.add_patch(r1)

    r2 = SphericalCircle(c0, 2*r200_deg*u.deg, resolution=100, vertex_unit='deg', 
                lw=3, label=r"$2 \times$ Cluster radius", edgecolor="red", facecolor="None", ls='--')
    ax1.add_patch(r2)

    ax1.set_title("Mock cluster {:d}".format(i_mock), fontdict=font)
    ax1.set_xlabel("R.A. (deg)", fontdict=font)
    ax1.set_ylabel("Dec. (deg)", fontdict=font)

    ax1.scatter(ra0, dec0, marker="X", s=100, color="red", label="Cluster center")

    ax1.tick_params(axis='x', rotation=45)
    ax1.invert_xaxis()

    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(sc, cax=cax1, orientation='vertical')
    cbar.set_label(label="Distance from center (deg)", fontdict=font)
    cbar.ax.tick_params(labelsize=labelsize)

    ax1.legend(fontsize=15)

    max_z = 0.6
    ax2 = fig.add_subplot(122)

    ax2.hist(z_spec[pre_mask][~halo_mask_restricted][dist_field < max_dist], bins=50, 
            color=colors1[7], histtype='stepfilled', edgecolor='k', range=[0, max_z], label="Field")[2]

    ax2.hist(halo_zspec, bins=50, color=colors1[3], histtype='stepfilled', edgecolor='k',
            range=[0, max_z], label="Cluster")[2]

    ax2.set_xlabel("z_spec", fontdict=font)
    ax2.set_ylabel("Number of objects", fontdict=font)

    # ax2.axvline(mean_z, color="red", lw=2, label="mean redshift")
    # ax2.axvline(median_z, color="yellow", lw=2, label="median redshift")
    ax2.axvline(z_cluster, color="yellow", lw=3, label="peak of redshift distribution")
    ax2.legend(fontsize=15, loc=2)

    fig.tight_layout(pad=3.0)
    plt.savefig("../figures/membership-testing-on-mocks/RA_DEC_z_distribution_field&cluster_mock{}.png".format(i_mock), dpi='figure', format='png')


    sigma_z = np.array([sigma_nmad_(r) for r in magr[pre_mask]])

    #This mask defines all galaxies in the fr*R200 projected area of the cluster
    mask_cluster_region = dist < fr*r200_deg

    #This one restricts the above mask to only the galaxies delimited by +/-fz*sigma from the cluster redshift, defining a list of candidate members
    mask_cluster = mask_cluster_region & (z_phot_pdfs[pre_mask] > z_cluster - fz1*sigma_z*(1+z_cluster)) & (z_phot_pdfs[pre_mask] < 
                                                                                                            z_cluster + fz1*sigma_z*(1+z_cluster))
    

    bw = (z_phot_pdfs[pre_mask][mask_cluster].max() - z_phot_pdfs[pre_mask][mask_cluster].min()) / z_bins

    kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(z_phot_pdfs[pre_mask][mask_cluster][:, np.newaxis])
    zz_phot = np.linspace(z_phot_pdfs[pre_mask][mask_cluster].min(), z_phot_pdfs[pre_mask][mask_cluster].max(), 
                        len(z_phot_pdfs[pre_mask][mask_cluster]))[:, np.newaxis]
    log_dens_zz_phot = kde.score_samples(zz_phot)

    zp_peak = zz_phot[np.argmax(np.exp(log_dens_zz_phot))]
    zp_bias = zp_peak - z_cluster

    z_phot_pdfs_unbiased = z_phot_pdfs - zp_bias

    mask_cluster_unbiased = mask_cluster_region & (z_phot_pdfs_unbiased[pre_mask] > z_cluster - 
                        fz1*sigma_z*(1+z_cluster)) & (z_phot_pdfs_unbiased[pre_mask] < z_cluster + fz1*sigma_z*(1+z_cluster))

    halo_zphot = z_phot_pdfs_unbiased[halo_mask]


    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(121)
    ax1.hist(z_phot_pdfs[pre_mask][mask_cluster], bins=z_bins, color=colors1[5], histtype='stepfilled', edgecolor='k', 
            range=[z_phot_pdfs[pre_mask][mask_cluster].min(), z_phot_pdfs[pre_mask][mask_cluster].max()], alpha=0.5, label="Photo-zs before unbiasing")[2]
    ax1.hist(z_phot_pdfs_unbiased[pre_mask][mask_cluster_unbiased], bins=z_bins, color=colors1[6], histtype='stepfilled', edgecolor='k', 
            range=[z_phot_pdfs_unbiased[pre_mask][mask_cluster_unbiased].min(), z_phot_pdfs_unbiased[pre_mask][mask_cluster_unbiased].max()], label="Photo-zs after unbiasing", alpha=0.5)[2]
    ax1.axvline(z_cluster, color="blue", lw=3, label="peak of spec-z distribution")

    ax1.set_title("Mock cluster {:d}".format(i_mock), fontdict=font)
    ax1.set_xlabel(r"$z_p$", fontdict=font)
    ax1.set_ylabel("Number of objects", fontdict=font)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(fontsize=labelsize)

    ax2 = fig.add_subplot(122)
    ax2.plot(zz_phot, np.exp(log_dens_zz_phot), lw=4, color=colors1[8], label="KDE estimation")
    ax2.axvline(z_cluster, color="blue", lw=3, label="peak of spec-z distribution")
    ax2.axvline(zp_peak, color="blueviolet", lw=3, label="peak of photo-z distribution before unbiasing")
    ax2.set_xlabel(r"$z_p$", fontdict=font)
    ax2.set_ylabel("Density of objects", fontdict=font)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(fontsize=labelsize)

    fig.tight_layout(pad=3.0)
    plt.savefig("../figures/membership-testing-on-mocks/zphot_biased_unbiased_mock{}.png".format(i_mock), dpi='figure', format='png')


    P_pz_C_array = P_pz_v2(z_cluster, magr[pre_mask][mask_cluster_unbiased],
                       means_pdfs_mock.T[pre_mask][mask_cluster_unbiased], 
                       weights_pdfs_mock.T[pre_mask][mask_cluster_unbiased],
                       stds_pdfs_mock.T[pre_mask][mask_cluster_unbiased], fz=fz2)
    

    mask_infalling = spec_members_vesc(m200, r200_mpc/cosmo.h, z_cluster, z_spec[pre_mask][mask_cluster_unbiased], 1.0)
    z_infalling = z_spec[pre_mask][mask_cluster_unbiased][mask_infalling]


    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(111)

    ax1.hist(z_infalling, bins=z_bins, histtype='stepfilled', color=colors3[7], alpha=1.0, edgecolor="black", linewidth=2.0)
    ax1.set_title(r"$|v_{pec}| < 1.0 \times v_{esc}$", fontdict=font)
    ax1.axvline(z_cluster, color="red", label=r"$z_C$")

    ax1.set_xlabel(r"$z$", fontdict=font)
    ax1.set_ylabel("Number of objects", fontdict=font)
    ax1.legend(fontsize=15, loc=2, framealpha=0.4)

    fig.tight_layout()
    plt.savefig("../figures/membership-testing-on-mocks/spec_members_mock{}.png".format(i_mock), dpi='figure', format='png')

    completeness_zp = []
    purity_zp = []
    d = 0.01
    Pmem_cut_arr_zp = np.arange(0.0, 1.0, d)

    for Pmem_cut in Pmem_cut_arr_zp:
        
        c = np.sum((P_pz_C_array > Pmem_cut) & mask_infalling) / np.sum(mask_infalling)
        p = np.sum((P_pz_C_array > Pmem_cut) & mask_infalling) / np.sum(P_pz_C_array > Pmem_cut)
        
        completeness_zp.append(c)
        purity_zp.append(p)

    #We need a criterium to determine a cut P(C | zp) > p0. One way is by optimizing the F1 score, defined below.
    completeness_zp = np.array(completeness_zp)
    purity_zp = np.array(purity_zp)

    F1_zp = 2 * completeness_zp*purity_zp/(completeness_zp + purity_zp)

    cut_zp = Pmem_cut_arr_zp[np.argmax(F1_zp)]


    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(111)

    ms = 10
    ax1.plot(Pmem_cut_arr_zp, completeness_zp, marker='o', color=colors1[2],
            label="Completeness", markersize=ms, markeredgecolor="black")
    ax1.plot(Pmem_cut_arr_zp, purity_zp, marker='o', color=colors1[5], 
            label="Purity", markersize=ms, markeredgecolor="black")
    ax1.plot(Pmem_cut_arr_zp, F1_zp, marker='o', color=colors1[8], 
            label="F1 score", markersize=ms, markeredgecolor="black")

    ax1.set_title("Mock cluster {:d}".format(i_mock), fontdict=font)
    ax1.set_ylabel("%", fontdict=font)
    ax1.set_xlabel("P(C|P(z)) cut", fontdict=font)

    ax1.axvline(cut_zp, color="black")
    ax1.text(x=cut_zp-0.18, y=0.75, s="Cut at {:.2f}".format(cut_zp), fontsize=15)

    ax1.legend(fontsize=labelsize, loc=2)
    plt.savefig("../figures/membership-testing-on-mocks/purity-completeness_P(C|zp)_(KNN-with-z)_fz{}_mock{}.png".format(fz2, i_mock), dpi='figure', 
                                                                                    format='png')
    

    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.scatter(ra[pre_mask][~halo_mask_restricted][dist_field < max_dist], dec[pre_mask][~halo_mask_restricted][dist_field < max_dist], 
                    c="lightblue", s=10, label="Field")
    ax1.scatter(ra[pre_mask][mask_cluster_unbiased][mask_infalling], dec[pre_mask][mask_cluster_unbiased][mask_infalling], c="blue", label="Spectroscopially nearby galaxies")
    ax1.set_title("Spectroscopically selected galaxies (Mock cluster {:d})".format(i_mock), fontsize=15)
    ax1.set_xlabel("R.A. (deg)", fontdict=font)
    ax1.set_ylabel("Dec. (deg)", fontdict=font)
    ax1.scatter(ra0, dec0, marker="X", s=100, color="red", label="Cluster center")
    ax1.legend(fontsize=labelsize)
    ax1.tick_params(axis='x', rotation=45)
    ax1.invert_xaxis()

    ax2.scatter(ra[pre_mask][~halo_mask_restricted][dist_field < max_dist], dec[pre_mask][~halo_mask_restricted][dist_field < max_dist], 
                    c="lightblue", s=10, label="Field")
    ax2.scatter(ra[pre_mask][mask_cluster_unbiased][P_pz_C_array > cut_zp], dec[pre_mask][mask_cluster_unbiased][P_pz_C_array > cut_zp], c="blue", 
                    label="Galaxies with P(C | zp) > {:.2f}".format(cut_zp))
    ax2.set_title("Galaxies selected with P(C | zp) cut", fontsize=15)
    ax2.set_xlabel("R.A. (deg)", fontdict=font)
    ax2.set_ylabel("Dec. (deg)", fontdict=font)
    ax2.scatter(ra0, dec0, marker="X", s=100, color="red", label="Cluster center")
    ax2.legend(fontsize=labelsize)
    ax2.tick_params(axis='x', rotation=45)
    ax2.invert_xaxis()

        
    fig.tight_layout(pad=3.0)
    plt.savefig("../figures/membership-testing-on-mocks/cluster_field_separation_P(C|zp)_(KNN-with-z)_fz{}_mock{}.png".format(fz2, i_mock), 
                                                                dpi='figure', format='png')
    

    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.hist(z_phot_pdfs_unbiased[pre_mask][mask_cluster_unbiased][mask_infalling], bins=z_bins, histtype='stepfilled', 
            color=colors3[1], alpha=0.5, label="photo-z of spec members (infalling)")
    ax1.hist(z_phot_pdfs_unbiased[pre_mask][mask_cluster_unbiased][P_pz_C_array > cut_zp], bins=z_bins, histtype='stepfilled', 
            color=colors3[4], alpha=0.5, label="photo-z of galaxies with P(C | zp) > {:.2f}".format(cut_zp))
    ax1.hist(z_phot_pdfs_unbiased[pre_mask][mask_cluster_unbiased][(P_pz_C_array > cut_zp) & mask_infalling], bins=z_bins, histtype='stepfilled', 
            color=colors3[8], alpha=0.4, label="photo-z of galaxies at the intersection")

    ax1.axvline(z_cluster - fz1*np.mean(sigma_z)*(1+z_cluster), color="red", ls='--')
    ax1.axvline(z_cluster + fz1*np.mean(sigma_z)*(1+z_cluster), color="red", ls='--', label=r"$z_C \pm \alpha \times mean(\sigma_{NMAD})$")
    ax1.axvline(z_cluster, color="red", label=r"$z_C$")

    ax1.set_title("Mock cluster {:d}".format(i_mock), fontdict=font)
    ax1.set_xlabel(r"$z_p$", fontdict=font)
    ax1.set_ylabel("Number of objects", fontdict=font)
    ax1.legend(fontsize=15, loc=2, framealpha=0.4)


    bins = 50
    ax2.hist(P_pz_C_array, bins=bins, histtype='stepfilled', 
            color=colors3[1], alpha=1.0)
    ax2.set_xlabel("P(C|zp)", fontdict=font)
    ax2.set_ylabel("Number of objects", fontdict=font)

    fig.tight_layout(pad=4.0)

    plt.savefig("../figures/membership-testing-on-mocks/zp_and_probability_distribution_(KNN-with-z)_fz{}_mock{}.png".format(fz2, i_mock), 
                dpi='figure', format='png')
    

    print("SAVING RESULTS")
    table[pre_mask][mask_cluster_unbiased][P_pz_C_array > cut_zp].write("tables/testing-on-mocks/membership-mock-clusters/cluster-mock_{}_members-{}fz1-{}fz2.csv".format(i_mock, fz1, fz2), 
                                                                        format='csv', overwrite=True)
    
    i_mock_list.append(i_mock)
    ra0_list.append(ra0)
    dec0_list.append(dec0)
    z_cluster_list.append(z_cluster)
    m200_list.append(m200.value) #solMass
    r200_mpc_list.append(r200_mpc.value[0]) #mpc
    r200_deg_list.append(r200_deg[0]) #deg
    zp_bias_list.append(zp_bias)
    c_zp_max_list.append(completeness_zp[np.argmax(F1_zp)])
    p_zp_max_list.append(purity_zp[np.argmax(F1_zp)])
    f1_zp_max_list.append(F1_zp[np.argmax(F1_zp)])
    c_zp_list.append(completeness_zp)
    p_zp_list.append(purity_zp)

    np.save("tables/testing-on-mocks/mock_clusters_properties/i_mock_list", i_mock_list)
    np.save("tables/testing-on-mocks/mock_clusters_properties/ra0_list", ra0_list)
    np.save("tables/testing-on-mocks/mock_clusters_properties/dec0_list", dec0_list)
    np.save("tables/testing-on-mocks/mock_clusters_properties/z_cluster_list", z_cluster_list)
    np.save("tables/testing-on-mocks/mock_clusters_properties/m200_list", m200_list)
    np.save("tables/testing-on-mocks/mock_clusters_properties/r200_mpc_list", r200_mpc_list)
    np.save("tables/testing-on-mocks/mock_clusters_properties/r200_deg_list", r200_deg_list)
    np.save("tables/testing-on-mocks/mock_clusters_properties/zp_bias_list", zp_bias_list)
    np.save("tables/testing-on-mocks/mock_clusters_properties/cut_zp_list", cut_zp_list)
    np.save("tables/testing-on-mocks/mock_clusters_properties/c_zp_max_list", c_zp_max_list)
    np.save("tables/testing-on-mocks/mock_clusters_properties/p_zp_max_list", p_zp_max_list)
    np.save("tables/testing-on-mocks/mock_clusters_properties/f1_zp_max_list", f1_zp_max_list)
    np.save("tables/testing-on-mocks/mock_clusters_properties/c_zp_list", c_zp_list)
    np.save("tables/testing-on-mocks/mock_clusters_properties/p_zp_list", p_zp_list)
