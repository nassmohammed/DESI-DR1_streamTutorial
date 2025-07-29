from scipy.interpolate import interp1d
from scipy.optimize import minimize
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import astropy.coordinates as coord
import pandas as pd

## every non-orbit function was written by Joseph Tang

def find_pole(lon1,lat1,lon2,lat2):
    """ Find the pole of a great circle orbit between two points.

    Parameters:
    -----------
    lon1 : longitude of the first point (deg)
    lat1 : latitude of the first point (deg)
    lon2 : longitude of the second point (deg)
    lat2 : latitude of the second point (deg)

    Returns:
    --------
    lon,lat : longitude and latitude of the pole
    """
    vec = np.cross(hp.ang2vec(lon1,lat1,lonlat=True),
                   hp.ang2vec(lon2,lat2,lonlat=True))
    lon,lat = hp.vec2ang(vec,lonlat=True)
    return [np.ndarray.item(lon),np.ndarray.item(lat)]

def ra_dec_to_phi1_phi2(frame, ra, dec):
    '''
    Given a frame, convert ra and dec to phi1 and phi2
    
    Input:
        frame: astropy.coordinates frame
        ra: right ascension in degrees
        dec: declination in degrees
        
    Output:
        phi1: stream phi1 coordinates in degrees
        phi2: stream phi2 coordinates in degrees
    '''
    skycoord_data = coord.SkyCoord(ra=ra, dec=dec, frame='icrs')
    transformed_skycoord = skycoord_data.transform_to(frame)
    phi1, phi2 = transformed_skycoord.phi1.deg, transformed_skycoord.phi2.deg
    return phi1, phi2  

def fit_orbit(stream_array, fr, progenitor_ra, fw, bw, theta, use_position=True):
    """
    Fit an orbit to the observed stream data.

    This function estimates the best-fit orbit parameters for a stellar stream by using
    the provided observational data and distance guess. 
    """

    #assuming no error in positions
    errs = [stream_array['PMRA_ERROR'], stream_array['PMDEC_ERROR'], stream_array['VRAD_ERR']]

    # Print each parameter with its label
    orbit_param_label = ["dec", "pmra", "pmdec", "vrad", "dist", "lsig_dec", "lsig_pmra", "lsig_pmdec", "lsig_vrad"]
    
    print('orbit parameters', orbit_param_label)
    print('intital guess', theta)
    
    optfunc = lambda theta: negloglike(theta, fr, stream_array, progenitor_ra, errs, bw, fw, use_position=use_position)
    results_o = minimize(optfunc, theta,  method="Powell")

    print("Optimization results_o:")
    print(f"Success: {results_o.success}")
    print(f"Status: {results_o.status}")
    print(f"Message: {results_o.message}")
    print(f"Function Value: {results_o.fun}")
    print(f"Number of Iterations: {results_o.nit}")
    print(f"Number of Function Evaluations: {results_o.nfev}")


    prog_distance = results_o.x[4] * u.kpc
    print(f'Progenitor')
    print(f'{orbit_param_label[0]}: {results_o.x[0]*u.deg:.2f}')
    print(f'{orbit_param_label[1]}: {results_o.x[1]*u.mas/u.yr:.2f}')
    print(f'{orbit_param_label[2]}: {results_o.x[2]*u.mas/u.yr:.2f}')
    print(f'{orbit_param_label[3]}: {results_o.x[3]*u.km/u.s:.2f}')
    print(f'{orbit_param_label[4]}: {results_o.x[4] * u.kpc:.2f}')
    print(f'{orbit_param_label[5]}: {10**results_o.x[5]*u.deg:.2f}')
    print(f'{orbit_param_label[6]}: {10**results_o.x[6]*u.mas/u.yr:.2f} or {(10**results_o.x[6]*u.mas/u.yr*prog_distance).to(u.km/u.s, equivalencies=u.dimensionless_angles()):.2f}')
    print(f'{orbit_param_label[7]}: {10**results_o.x[7]*u.mas/u.yr:.2f} or {(10**results_o.x[7]*u.mas/u.yr*prog_distance).to(u.km/u.s, equivalencies=u.dimensionless_angles()):.2f}')
    print(f'{orbit_param_label[8]}: {10**results_o.x[8]*u.km/u.s:.2f}')
    #print peri and apo
    orbit = orbit_model(results_o.x[0:5], progenitor_ra, bw, fw, return_o=True)[6]
    print(f'Pericenter: {orbit.rperi()*u.kpc:.2f} ')
    print(f'Apocenter: {orbit.rap()*u.kpc:.2f}')
    period = orbit.Tr()  # in galpy's default time units


    print(f"Orbital period: {period:.2f}")

    return results_o, orbit


def negloglike(theta, fr, stream_array, ra_prog, param_errs, ts_rw, ts_ff, use_position=True):
    stream_ra, stream_dec, stream_pmra, stream_pmdec, stream_vrad = stream_array['RA'], stream_array['DEC'], stream_array['PMRA'], stream_array['PMDEC'], stream_array['VRAD']
    pmra_err, pmdec_err, vrad_err = param_errs #observed error

    o1_model_ra, o1_model_dec, o1_model_pmra, o1_model_pmdec, o1_model_vlos, o1_model_dist = np.asarray(orbit_model(theta[0:5], ra_prog, ts_rw, ts_ff))
    lsig_phi2, lsig_pmra, lsig_pmdec, lsig_vrad = theta[5:] #log sigmas

    stream_phi1, stream_phi2 = ra_dec_to_phi1_phi2(fr, np.asarray(stream_ra)*u.deg, np.asarray(stream_dec)*u.deg)
    o1_model_phi1, o1_model_phi2 = ra_dec_to_phi1_phi2(fr, o1_model_ra*u.deg, o1_model_dec*u.deg)

    phi2_y = interp1d(o1_model_phi1, o1_model_phi2, kind='linear', fill_value='extrapolate')
    pmra_y = interp1d(o1_model_phi1, o1_model_pmra, kind='linear', fill_value='extrapolate')
    pmdec_y = interp1d(o1_model_phi1, o1_model_pmdec, kind='linear', fill_value='extrapolate')
    vlos_y = interp1d(o1_model_phi1, o1_model_vlos, kind='linear', fill_value='extrapolate')


    resid_phi2 = residuals(phi2_y, stream_phi2, stream_phi1)
    resid_pmra = residuals(pmra_y, stream_pmra, stream_phi1)
    resid_pmdec = residuals(pmdec_y, stream_pmdec, stream_phi1)
    resid_vlos = residuals(vlos_y, stream_vrad, stream_phi1)

    phi2_err = 0
    chi2_phi2 = resid_phi2**2/(phi2_err**2 + (10**lsig_phi2)**2)
    const_phi2 = np.log(2 * np.pi * (phi2_err**2+(10**lsig_phi2)**2))
    logl_phi2 = -0.5*np.sum(chi2_phi2 + const_phi2)

    chi2_pmra = resid_pmra**2/(pmra_err**2+ (10**lsig_pmra)**2)
    const_pmra = np.log(2 * np.pi * (pmra_err**2+(10**lsig_pmra)**2))
    logl_pmra = -0.5*np.sum(chi2_pmra + const_pmra)

    chi2_pmdec = resid_pmdec**2/(pmdec_err**2+(10**lsig_pmdec)**2)
    const_pmdec = np.log(2 * np.pi * (pmdec_err**2+(10**lsig_pmdec)**2))
    logl_pmdec = -0.5*np.sum(chi2_pmdec + const_pmdec)

    chi2_vlos = resid_vlos**2/(vrad_err**2+(10**lsig_vrad)**2)
    const_vlos = np.log(2 * np.pi * (vrad_err**2+(10**lsig_vrad)**2))
    logl_vlos = -0.5*np.sum(chi2_vlos + const_vlos)

    if use_position:
        neg_logl_tot = -logl_phi2 - logl_pmra - logl_pmdec - logl_vlos
    else:
        neg_logl_tot = -logl_pmra - logl_pmdec - logl_vlos
    return neg_logl_tot

def residuals(o1_interp_func, stream_val, stream_phi1):
    return np.abs(o1_interp_func(stream_phi1) - stream_val)

def orbit_model(theta, ra_prog, ts_rw, ts_ff, values=True, return_o=False):
    ra = ra_prog
    dec, pmra, pmdec, vrad, dist = theta[0:5]

    ra = ra*u.deg
    dec = dec*u.deg
    dist = dist*u.kpc
    pmra_cosdec = pmra*u.mas/u.yr*np.cos(dec)
    pmdec = pmdec*u.mas/u.yr
    vrad = vrad*u.km/u.s

    o_rw = Orbit(vxvv=[ra, dec, dist, pmra_cosdec, pmdec, vrad], radec=True)
    o_rw.integrate(ts_rw, MWPotential2014)
    model_ra_rw, model_dec_rw, model_pmra_rw, model_pmdec_rw, model_vlos_rw, model_dist_rw = o_rw.ra(ts_rw), o_rw.dec(ts_rw), o_rw.pmra(ts_rw), o_rw.pmdec(ts_rw), o_rw.vlos(ts_rw), o_rw.dist(ts_rw) ##!!!! take array of distance

    o_ff = Orbit(vxvv=[ra, dec, dist, pmra_cosdec, pmdec, vrad], radec=True)
    o_ff.integrate(ts_ff, MWPotential2014)
    model_ra_ff, model_dec_ff, model_pmra_ff, model_pmdec_ff, model_vlos_ff, model_dist_ff = o_ff.ra(ts_ff), o_ff.dec(ts_ff), o_ff.pmra(ts_ff), o_ff.pmdec(ts_ff), o_ff.vlos(ts_ff), o_ff.dist(ts_ff) ##!!!! take array of distance

    model_ra = np.concatenate([model_ra_rw[::-1], model_ra_ff])
    model_dec = np.concatenate([model_dec_rw[::-1], model_dec_ff])
    model_pmra = np.concatenate([model_pmra_rw[::-1], model_pmra_ff])
    model_pmdec = np.concatenate([model_pmdec_rw[::-1], model_pmdec_ff])
    model_vlos = np.concatenate([model_vlos_rw[::-1], model_vlos_ff])
    model_dist = np.concatenate([model_dist_rw[::-1], model_dist_ff])
    if return_o:
        return model_ra, model_dec, model_pmra, model_pmdec, model_vlos, model_dist, o_rw
    else:
        return model_ra, model_dec, model_pmra, model_pmdec, model_vlos, model_dist
def vhel_to_vgsr(data_ra, data_dec, data_vhel):
    '''
    Convert heliocentric velocities to Galactocentric velocities.
    
    Input:
        vhel (array): Array of heliocentric radial velocities. [km/s]
        ra (array): Array of right ascension values. [deg]
        dec (array): Array of declination values. [deg]
        
    Output:
        vgsr (array): Array of Galactocentric radial velocities
    '''
    if not isinstance(data_vhel, u.quantity.Quantity):
        data_vhel = data_vhel * u.km/u.s
        
    if not isinstance(data_ra, u.quantity.Quantity):
        data_ra = data_ra * u.deg
        
    if not isinstance(data_dec, u.quantity.Quantity):
        data_dec = data_dec * u.deg
        
    icrs = coord.SkyCoord(ra=data_ra, dec=data_dec, radial_velocity=data_vhel, frame='icrs')
    data_vgsr = rv_to_gsr(icrs)
        
    return data_vgsr

def rv_to_gsr(c, v_sun=None):
    """Transform a barycentric radial velocity to the Galactic Standard of Rest
    (GSR).

    The input radial velocity must be passed in as a

    Parameters
    ----------
    c : `~astropy.coordinates.BaseCoordinateFrame` subclass instance
        The radial velocity, associated with a sky coordinates, to be
        transformed.
    v_sun : `~astropy.units.Quantity`, optional
        The 3D velocity of the solar system barycenter in the GSR frame.
        Defaults to the same solar motion as in the
        `~astropy.coordinates.Galactocentric` frame.

    Returns
    -------
    v_gsr : `~astropy.units.Quantity`
        The input radial velocity transformed to a GSR frame.

    """
    if v_sun is None:
        v_sun = coord.Galactocentric().galcen_v_sun.to_cartesian()

    gal = c.transform_to(coord.Galactic)
    cart_data = gal.data.to_cartesian()
    unit_vector = cart_data / cart_data.norm()

    v_proj = v_sun.dot(unit_vector)

    return c.radial_velocity + v_proj

def plot_orbit(model_ra, model_dec, stream_ra, stream_dec, ra_prog):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="mollweide")

    # Convert RA to radians and shift to [-180, 180] for Mollweide projection
    model_ra_rad = np.radians(zero_360_to_180(model_ra))
    stream_ra_rad = np.radians(zero_360_to_180(stream_ra))
    ra_prog_rad = np.radians(zero_360_to_180(np.array([ra_prog])))

    # Convert Dec to radians
    model_dec_rad = np.radians(model_dec)
    stream_dec_rad = np.radians(stream_dec)
    dec_prog_rad = np.radians(interp1d(model_ra, model_dec, kind='linear', fill_value='extrapolate')(ra_prog))

    ax.scatter(model_ra_rad, model_dec_rad, c='r', s=0.5, zorder=0)
    legend_handles = [plt.Line2D([],[], c='r', lw=1, label='Model Orbit')]
    legend_handles.append(plt.Line2D([],[], c='k', linestyle='None', marker='o', label='Stream Data', markersize=2))
    #legend_handles.append(plt.Line2D([],[], c='white', marker='*', label='Progenitor Location', markersize=10, markeredgecolor='black'))
    ax.scatter(stream_ra_rad, stream_dec_rad, c='k', s=10, label='Stream Data')
    ax.scatter(ra_prog_rad, dec_prog_rad, marker='*', label='Progenitor Location', s=100, facecolor='white', edgecolor='black')

    ax.set_xlabel(r'$\alpha$ [deg]')
    ax.set_ylabel(r'$\delta$ [deg]')
    ax.legend(handles = legend_handles, loc='upper right', fontsize=10)
    return fig, ax

def zero_360_to_180(ra):
    ra_copy = np.copy(ra)
    where_180 = np.where(ra_copy > 180)
    ra_copy[where_180] = ra_copy[where_180] - 360
        
    return ra_copy

def orbit_interpolations(o_s):
    o_phi1, o_phi2, o_ra, o_dec, o_pmra, o_pmdec, o_vrad, o_vgsr, o_dist = o_s
    ointerp_phi1 = interp1d(o_phi1, o_phi1, kind='linear', fill_value='extrapolate')
    ointerp_phi2 = interp1d(o_phi1, o_phi2, kind='linear', fill_value='extrapolate')
    ointerp_ra = interp1d(o_phi1, o_ra, kind='linear', fill_value='extrapolate')
    ointerp_dec = interp1d(o_phi1, o_dec, kind='linear', fill_value='extrapolate')
    ointerp_pmra = interp1d(o_phi1, o_pmra, kind='linear', fill_value='extrapolate')
    ointerp_pmdec = interp1d(o_phi1, o_pmdec, kind='linear', fill_value='extrapolate')
    ointerp_vrad = interp1d(o_phi1, o_vrad, kind='linear', fill_value='extrapolate')
    ointerp_vgsr = interp1d(o_phi1, o_vgsr, kind='linear', fill_value='extrapolate')
    ointerp_dist = interp1d(o_phi1, o_dist, kind='linear', fill_value='extrapolate')
    df = {
        "phi1": ointerp_phi1,
        "phi2": ointerp_phi2,
        "ra": ointerp_ra,
        "dec": ointerp_dec,
        "pmra": ointerp_pmra,
        "pmdec": ointerp_pmdec,
        "vrad": ointerp_vrad,
        "vgsr": ointerp_vgsr,
        "dist": ointerp_dist
    }
    return df