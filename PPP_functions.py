import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline


def angular_distance(RA, DEC, RA0, DEC0):
    """
    Calculate angular distance between points (RA, DEC) and reference point (RA0, DEC0).
    
    Uses the Haversine formula for great-circle distance.
    
    Parameters
    ----------
    RA, DEC : array_like
        Coordinates of points (degrees)
    RA0, DEC0 : float
        Reference point coordinates (degrees)
    
    Returns
    -------
    array_like
        Angular distances in degrees
    """
    ra = np.radians(RA)
    dec = np.radians(DEC)
    ra0 = np.radians(RA0)
    dec0 = np.radians(DEC0)
    
    dra = (ra - ra0 + np.pi) % (2*np.pi) - np.pi
    ddec = dec - dec0
    
    a = np.sin(ddec/2)**2 + np.cos(dec) * np.cos(dec0) * np.sin(dra/2)**2
    a = np.clip(a, 0, 1)
    
    return 2 * np.arcsin(np.sqrt(a)) * 180/np.pi


def elliptical_radius(RA, DEC, RA0, DEC0, q, phi):
    """
    Calculate elliptical radius from center for points with given ellipticity.
    
    Parameters
    ----------
    RA, DEC : array_like
        Coordinates of points (degrees)
    RA0, DEC0 : float
        Center coordinates (degrees)
    q : float
        Ellipticity ratio (b/a, 0 < q <= 1)
    phi : float
        Position angle of ellipse major axis (radians)
    
    Returns
    -------
    array_like
        Elliptical radii in degrees
    """
    ra = np.radians(RA)
    dec = np.radians(DEC)
    ra0 = np.radians(RA0)
    dec0 = np.radians(DEC0)
    
    dra = (ra - ra0 + np.pi) % (2*np.pi) - np.pi
    
    x = dra * np.cos(dec0) * 180/np.pi
    y = (dec - dec0) * 180/np.pi
    
    xp =  x * np.cos(phi) + y * np.sin(phi)
    yp = -x * np.sin(phi) + y * np.cos(phi)
    
    return np.sqrt(xp**2 + (yp/q)**2)

# Alias for convenience
elliptical_radius_xy = elliptical_radius


import warnings
warnings.filterwarnings('ignore')

def NFW_profile(R, rs, rho_s, Rt, tau):
    '''
    Lokas & Moman 2001 project NFW profile with truncation 
    '''

    R = np.radians(R)
    rs = np.radians(rs)
    Rt = np.radians(Rt)
    
    x = R/rs

    C_gt1 = np.where(x > 1, np.arccos(1/x), 0)
    C_lt1 = np.where(x < 1, np.arccosh(1/x), 0)
    C = C_gt1 + C_lt1
        
    Sigma_c = 2*rs*rho_s/(x**2 - 1) * (1 - C/np.sqrt(np.abs(x**2 - 1))) * np.exp(-(R/Rt)**tau)
    
    return Sigma_c


def norm_cluster_elliptical(p, model, Rmax, tau_fixed=False, tau_fixed_value=None):
    """
    Calculate normalization factor for elliptical cluster profile.
    
    The singularity at R=0 is handled by setting f(0) = 0.
    
    Parameters
    ----------
    rs : float
        Scale radius of the cluster (degrees)
    rho_s : float
        Central density of the cluster
    Rt : float
        Truncation radius (degrees)
    q : float
        Ellipticity ratio (b/a, 0 < q <= 1)
    tau : float
        Truncation exponent for outer profile
    Rmax : float
        Maximum radius for integration (degrees)
    
    Returns
    -------
    float
        Normalization constant (2πq times the radial integral)
    """

    # Adaptive grid: more points where the profile changes rapidly
    Rg = np.linspace(0, Rmax, 2000)
    Rg[0] = 1e-8

    if tau_fixed:
        if model == "NFW": rs, rho_s, Rt, q = p
        else: Rc, Rt, q, alpha = p
        tau = tau_fixed_value
        
    else:
        if model == "NFW": rs, rho_s, Rt, tau, q = p
        else: Rc, Rt, q, alpha, tau = p

    # Adaptive grid: more points where the profile changes rapidly
    Rg = np.linspace(0, Rmax, 2000)
    Rg[0] = 1e-8
    
    if model == "NFW":
        f = NFW_profile(Rg, rs, rho_s, Rt, tau)
        f[0] = 0.0

    elif model == "Sersic":
        f = f = np.exp(-(Rg/Rc)**alpha) * np.exp(-(Rg/Rt)**tau)

    elif model == "Cored":
        f = (1 + Rg/Rc)**(-alpha) * np.exp(-(Rg/Rt)**tau)
    
    integrand = f * Rg
    dR = np.diff(Rg)
    
    # Trapezoidal rule integration
    val = np.sum(0.5 * (integrand[1:] + integrand[:-1]) * dR)
    
    return 2 * np.pi * q * val


def NFW_density(params, ra, dec, ra0, dec0, Rmax, Re, tau_fixed=False, tau_fixed_value=None):

    if tau_fixed:
        rs, rho_s, Rt, q, phi, Sigma_b = params
        tau = tau_fixed_value
    else:
        rs, rho_s, Rt, tau, q, phi, Sigma_b = params

    # --- Compute Nc ---
    R = angular_distance(ra, dec, ra0, dec0)
    N_inner = np.sum(R <= Rmax/2)
    area_inner = np.pi * (Rmax/2)**2
    N_bg_expected_inner = Sigma_b * area_inner
    Nc = max(1, N_inner - N_bg_expected_inner)
    
    # --- Fitted model ---
    norm_fit = norm_cluster_elliptical([rs, rho_s, Rt, q], "NFW", Rmax, tau_fixed=tau_fixed, tau_fixed_value=tau_fixed_value)
    C_fit = Nc / norm_fit
    
    # Truncated Sersic profile
    Sigma_c_fit = C_fit * NFW_profile(Re, rs, rho_s, Rt, tau)
    model_fit = Sigma_c_fit + Sigma_b

    return model_fit


def Sersic_density(params, ra, dec, ra0, dec0, Rmax, Re, tau_fixed=False, tau_fixed_value=None):

    if tau_fixed:
        Rc, Rt, q, phi, alpha, Sigma_b = params
        tau = tau_fixed_value
    else:
        Rc, Rt, q, phi, alpha, tau, Sigma_b = params

    # --- Compute Nc ---
    R = angular_distance(ra, dec, ra0, dec0)
    N_inner = np.sum(R <= Rmax/2)
    area_inner = np.pi * (Rmax/2)**2
    N_bg_expected_inner = Sigma_b * area_inner
    Nc = max(1, N_inner - N_bg_expected_inner)
    
    # --- Fitted model ---
    norm_fit = norm_cluster_elliptical([Rc, Rt, q, alpha], "Sersic", Rmax, tau_fixed=tau_fixed, tau_fixed_value=tau_fixed_value)
    C_fit = Nc / norm_fit
    
    # Truncated Sersic profile
    Sigma_c_fit = C_fit * np.exp(-(Re/Rc)**alpha) * np.exp(-(Re/Rt)**tau)
    model_fit = Sigma_c_fit + Sigma_b

    return model_fit


def Cored_density(params, ra, dec, ra0, dec0, Rmax, Re, tau_fixed=False, tau_fixed_value=None):

    if tau_fixed:
        Rc, Rt, q, phi, alpha, Sigma_b = params
        tau = tau_fixed_value
    else:
        Rc, Rt, q, phi, alpha, tau, Sigma_b = params

    # --- Compute Nc ---
    R = angular_distance(ra, dec, ra0, dec0)
    N_inner = np.sum(R <= Rmax/2)
    area_inner = np.pi * (Rmax/2)**2
    N_bg_expected_inner = Sigma_b * area_inner
    Nc = max(1, N_inner - N_bg_expected_inner)
    
    # --- Fitted model ---
    norm_fit = norm_cluster_elliptical([Rc, Rt, q, alpha], "Cored", Rmax, tau_fixed, tau_fixed_value)
    C_fit = Nc / norm_fit
    
    # Truncated Sersic profile
    Sigma_c_fit = C_fit * (1 + Re/Rc)**(-alpha) * np.exp(-(Re/Rt)**tau)
    model_fit = Sigma_c_fit + Sigma_b

    return model_fit


def splashback_radius(params, model, ra, dec, ra0, dec0, Rmax, Re, tau_fixed=False, tau_fixed_value=None):
    '''
    Compute the 2D splashback radius (Rsp)
    The 3D r_sp can be obtained by r_sp = sqrt(pi/2) * Rsp (More et al. 2016, see also Lucas's paper (Gabriel-Silva & Sodré 2025))
    '''
    
    Sigma_ = model(params, ra, dec, ra0, dec0, Rmax, Re, tau_fixed, tau_fixed_value)


    #This is like doing dlog(Sigma_)/dlog(R) = R/Sigma_ * dSigma_/dR
    #s is for smoothing the function. The final result depends on the value of s, but for lack of further information we set s=None.
    Sigma_spline = UnivariateSpline(Re, Sigma_, s=None)
    gamma = Re/Sigma_ * Sigma_spline.derivative()(Re)

    #I am limiting the search for the minimum of gamma to R < 4*r200, 
    #because after that the profile is dominated by the background and it creates a false minimum in gamma
    Rsp = Re[np.argmin(gamma)]

    return Rsp, gamma


def plot_radial_adapted(ax, color, params, model, RA, DEC, RA0, DEC0, Rmax, tau_fixed=False, tau_fixed_value=None, nbins=20):
    """
    Plot elliptical radial profile with fitted model overlaid.
    
    Parameters
    ----------
    ax : Axes
        Axes object. Can be a single figure or an element of a subplots axis ndarray.
    RA, DEC : array_like
        Coordinates of observed galaxies (degrees)
    fit : dict
        Fitted model dictionary from fit_cluster_elliptical
    RA0, DEC0 : float
        Cluster center coordinates (degrees)
    Rmax : float
        Maximum radius for plotting (degrees)
    nbins : int
        Number of radial bins
    """

    if model == "NFW":
        rs_fit, rho_s_fit, Rt_fit, q_fit, phi_fit, Sigma_b = params
    else:
        Rc_fit, Rt_fit, q_fit, phi_fit, alpha_fit, Sigma_b = params
    
    Re = elliptical_radius_xy(RA, DEC, RA0, DEC0, q_fit, phi_fit)
    
    # Remove non-finite values AND values beyond Rmax
    mask = np.isfinite(Re) & (Re <= Rmax)
    Re = Re[mask]
    
    # Use Rmax for binning
    breaks = np.linspace(0, Rmax, nbins + 1)
    
    counts, _ = np.histogram(Re, bins=breaks)
    Rmid = (breaks[:-1] + breaks[1:]) / 2
    
    areas = np.pi * q_fit * (breaks[1:]**2 - breaks[:-1]**2)
    
    density = counts / areas
    err = np.sqrt(np.maximum(counts, 1)) / areas
    
    # --- Compute Nc ---
    R = angular_distance(RA, DEC, RA0, DEC0)
    N_inner = np.sum(R <= Rmax/2)
    area_inner = np.pi * (Rmax/2)**2
    N_bg_expected_inner = Sigma_b * area_inner
    Nc = max(1, N_inner - N_bg_expected_inner)
    
    # --- Fitted model ---
    if model == "NFW":
        norm_fit = norm_cluster_elliptical([rs_fit, rho_s_fit, Rt_fit, q_fit], model, Rmax, tau_fixed=tau_fixed, tau_fixed_value=tau_fixed_value)
    else:
        norm_fit = norm_cluster_elliptical([Rc_fit, Rt_fit, q_fit, alpha_fit], model, Rmax, tau_fixed=tau_fixed, tau_fixed_value=tau_fixed_value)
    C_fit = Nc / norm_fit
    
    Rg = np.linspace(0, Rmax, 200)
    
    if model == "NFW":
        # Truncated NFW-like profile
        # Handle singularity at R=0: model diverges, set to a large finite value for plotting
        with np.errstate(divide='ignore', invalid='ignore'):
            Sigma_c_fit = C_fit * NFW_profile(Rg, rs_fit, rho_s_fit, Rt_fit, tau_fixed_value)
        Sigma_c_fit[~np.isfinite(Sigma_c_fit)] = np.nan

    elif model == "Sersic":
        Sigma_c_fit = C_fit * np.exp(-(Rg/Rc_fit)**alpha_fit) * np.exp(-(Rg/Rt_fit)**tau_fixed_value)

    elif model == "Cored":
        Sigma_c_fit = C_fit * (1 + Rg/Rc_fit)**(-alpha_fit) * np.exp(-(Rg/Rt_fit)**tau_fixed_value)

    else:
        print("Invalid profile")

    model_fit = Sigma_c_fit + Sigma_b
    
    # --- Plot ---
    ax.errorbar(Rmid/Rmax, density, yerr=err, fmt='o', color=color, capsize=3)
    
    ax.plot(Rg/Rmax, model_fit, color, linewidth=2, label="{}".format(model))
    ax.axhline(y=Sigma_b, xmin=0, xmax=1.0, color='blue', linewidth=2, linestyle='--')
    
    # Set y-limit excluding NaN values from the cusp
    ylim_max = np.nanmax(np.concatenate([density + err, model_fit]))
    ax.set_xlim(0, 1)
    ax.set_ylim(Sigma_b - 40, ylim_max * 1.2)

    ax.set_yscale("log")