from arrow import get
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy import table
import stream_functions as stream_funcs
import pandas as pd
from matplotlib.colors import Normalize
import galstreams
import astropy.units as u
import astropy.coordinates as coord
from gala.coordinates import GreatCircleICRSFrame
import importlib
from astropy.table import Table
import scipy as sp
from collections import OrderedDict
import gallery_functions as gallery_funcs
importlib.reload(gallery_funcs)
from stream_functions import apply_spline
import vdisp
import os
import corner

def zero_360_to_180(ra):
    ra_copy = np.copy(ra)
    where_180 = np.where(ra_copy > 180)
    ra_copy[where_180] = ra_copy[where_180] - 360
        
    return ra_copy
def get_mem_path(directory):
    import os
    files = os.listdir(directory)
    for file in files:
        if '0.5%_mem' in file and file.endswith('.fits'):
            return os.path.join(directory, file)

param_labels = ["lpstream",
                "v1","v2","v3","lsigv",
                "feh1","lsigfeh",
                "pmra1","pmra2","pmra3","lsigpmra",
                "pmdec1","pmdec2","pmdec3","lsigpmdec",
                "bv", "lsigbv", "bfeh", "lsigbfeh", "bpmra", "lsigbpmra", "bpmdec", "lsigbpmdec"]


def get_paramdict(theta, labels = param_labels):
    '''Make an ordered dictionary of the parameters as keys and inputted theta as values'''
    return OrderedDict(zip(labels, theta))

import orbit_functions as ofuncs
def plot_form(ax):
    ax.grid(ls='-.', alpha=0.2, zorder=0)
    ax.tick_params(direction='in')
    ax.spines['top'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax.minorticks_on()

def process_chain(chain, avg_error=True, labels=param_labels):
    ''' Returns the means and errors of teh parameters
    
    Parameters:
    chain - array. The chain
    avg_error - bool. Will average the + and - errors if True
    
    Return:
    2 or 3 OrderedDict as a tuple. means, errors or means, errors+, errors-
    '''
    pctl = np.percentile(chain, [16, 50, 84], axis=0)
    meds = pctl[1]
    ep = pctl[2]-pctl[1]
    em = pctl[0]-pctl[1]

    if avg_error: # just for simplicity, assuming no asymmetry
        err = (ep-em)/2
        return OrderedDict(zip(labels, meds)), OrderedDict(zip(labels, err))
    else:
        return OrderedDict(zip(labels, meds)), OrderedDict(zip(labels, ep)), OrderedDict(zip(labels, em))
def plx_mask(min_dist, plx, plx_err):
    return (plx - 2*plx_err) < 1/min_dist

def d2dm(d):
    '''d in kpc'''
    return 5*np.log10(d*1000)-5

def load_stream_mems(mem_path = '/home/jupyter-nassermoha/raid_nassermoha/data/runs/C-19-I21_250529_final_const/C-19-I21_phi2_spline_0.5%_mem.fits'):

    stream_run_directory =  os.path.dirname(mem_path)
    stream_data = table.Table.read(mem_path)
    return stream_data, stream_run_directory

def load_desi_data(desi_path= '/raid/DESI/catalogs/loa/rv_output/241119/rvpix-loa.fits',
                  distance_path='/raid/DESI/catalogs/loa/rv_output/241119/rvsdistnn-loa-241126.fits',
                  decals_path='/raid/DESI/catalogs/loa/rv_output/241119/legacyphot-loa-241126.fits',
                  desired_columns=None, fr=None, local=False):
    """
    Joseph's code to load DESI data
    """
    if not local:
        desired_columns = [
        'VRAD', 'VRAD_ERR', 'RVS_WARN', 'PARALLAX', 'PARALLAX_ERROR', 
        'RR_SPECTYPE', 'PMRA', 'PMRA_ERROR', 'PMDEC', 'PMDEC_ERROR', 
        'TARGET_RA', 'TARGET_DEC', 'FEH', 'FEH_ERR', 'SOURCE_ID', 
        'TARGETID', 'PMRA_PMDEC_CORR', 'PRIMARY'
    ]
        desi_hdu_indices = [1, 4]
        desi_vrad_data = stream_funcs.load_fits_columns(desi_path, desired_columns, desi_hdu_indices)

        # Load Sergey's distance data
        dist_columns = ['TARGETID', 'dist_mod', 'dist_mod_err']
        distance_data  = stream_funcs.load_fits_columns(distance_path, dist_columns)

        # Load the DECaLS data
        decal_columns = ['EBV', 'FLUX_G', 'FLUX_R']
        decal_data = stream_funcs.load_fits_columns(decals_path, decal_columns)

        # Combine the data
        desi_data = table.hstack([desi_vrad_data, distance_data, decal_data])
        del desi_vrad_data, distance_data, decal_data

        # Delete the repeated TargetID column
        if len(np.where(desi_data['TARGETID_1'].value == desi_data['TARGETID_2'].value)[0]) == len(desi_data):
            desi_data.remove_columns(['TARGETID_2'])
            desi_data.rename_column('TARGETID_1', 'TARGETID')
            
        elif len(np.where(desi_data['TARGETID_1'].value == desi_data['TARGETID_2'].value)[0]) != len(desi_data):
            print('The TargetID columns do not match')


        # Drop the rows with NaN values in all columns
        print(f"Length of DESI Data before Cuts: {len(desi_data)}")
        drop_nan_columns = np.concatenate((desired_columns, decal_columns))
        desi_dropped_nan_df = stream_funcs.dropna_Table(desi_data, columns = drop_nan_columns) # Custom function to drop rows with NaN values
        print(f"Length of DESI Data after NaN cut: {len(desi_dropped_nan_df)}")

        # Drop the rows with 'RVS_WARN' != 0 and 'RR_SPECTYPE' != 'STAR', are not duplicates, and with low enough radial velocity and metallicity errors
        desi_dropped_vals = desi_dropped_nan_df[(desi_dropped_nan_df['RVS_WARN'] == 0) & (desi_dropped_nan_df['RR_SPECTYPE'] == 'STAR') & (desi_dropped_nan_df['PRIMARY']) &\
            (desi_dropped_nan_df['VRAD_ERR'] < 10) & (desi_dropped_nan_df['FEH_ERR'] < 0.5)]
        
        # Drop the rows with 'RVS_WARN' != 0 and 'RR_SPECTYPE' != 'STAR', are not duplicates, and with low enough radial velocity and metallicity errors
        sel_qual = (desi_dropped_nan_df['RVS_WARN'] == 0) & (desi_dropped_nan_df['RR_SPECTYPE'] == 'STAR') & (desi_dropped_nan_df['PRIMARY']) &\
            (desi_dropped_nan_df['VRAD_ERR'] < 10) & (desi_dropped_nan_df['FEH_ERR'] < 0.5)


        print(f"Length of DESI data after RVS_WARN, RR_SPECTYPE, PRIMARY, VRAD_ERR, and FEH_ERR: {len(desi_dropped_vals)}")

        # Drop the columns 'RVS_WARN' and 'RR_SPECTYPE' and convert to pandas DataFrame
        desi_dropped_vals.remove_columns(['RVS_WARN', 'RR_SPECTYPE'])
        desi_dropped_vals = desi_dropped_vals.to_pandas()

        # Add a floor to the uncertainties since they are underestimated
        desi_dropped_vals['VRAD_ERR'] = np.sqrt(desi_dropped_vals['VRAD_ERR']**2 + 0.9**2) ### Turn into its own column
        desi_dropped_vals['PMRA_ERROR'] = np.sqrt(desi_dropped_vals['PMRA_ERROR']**2 + (np.sqrt(550)*0.001)**2) ### Turn into its own column
        desi_dropped_vals['PMDEC_ERROR'] = np.sqrt(desi_dropped_vals['PMDEC_ERROR']**2 + (np.sqrt(550)*0.001)**2) ### Turn into its own column
        desi_dropped_vals['FEH_ERR'] = np.sqrt(desi_dropped_vals['FEH_ERR']**2 + 0.01**2) ### Turn into its own column

        # Delete some old variables
        del desi_dropped_nan_df, desi_data

        desi_data = desi_dropped_vals

        del desi_dropped_vals
        print('converting to phi1, phi2...')
        desi_data.loc[:,'phi1'], desi_data.loc[:,'phi2']  = stream_funcs.ra_dec_to_phi1_phi2(fr, np.array(desi_data['TARGET_RA'])*u.deg, np.array(desi_data['TARGET_DEC'])*u.deg)
        desi_data['VGSR'] =  np.array(stream_funcs.vhel_to_vgsr(np.array(desi_data['TARGET_RA'])*u.deg, np.array(desi_data['TARGET_DEC'])*u.deg, np.array(desi_data['VRAD'])*u.km/u.s).value)
        desi_data['VGSR_ERR'] = desi_data['VRAD_ERR']
        return desi_data
    else:
         desi_data_tbl = table.Table.read(desi_path, format='fits')
         desi_data = pd.DataFrame(desi_data_tbl.as_array())

def get_sel_qual_mask(
    desi_path='/raid/DESI/catalogs/loa/rv_output/241119/rvpix-loa.fits',
    distance_path='/raid/DESI/catalogs/loa/rv_output/241119/rvsdistnn-loa-241126.fits',
    decals_path='/raid/DESI/catalogs/loa/rv_output/241119/legacyphot-loa-241126.fits'
):
    desired_columns = [
        'VRAD', 'VRAD_ERR', 'RVS_WARN', 'PARALLAX', 'PARALLAX_ERROR', 
        'RR_SPECTYPE', 'PMRA', 'PMRA_ERROR', 'PMDEC', 'PMDEC_ERROR', 
        'TARGET_RA', 'TARGET_DEC', 'FEH', 'FEH_ERR', 'SOURCE_ID', 
        'TARGETID', 'PMRA_PMDEC_CORR', 'PRIMARY'
    ]
    decal_columns = ['EBV', 'FLUX_G', 'FLUX_R']
    dist_columns = ['TARGETID', 'dist_mod', 'dist_mod_err']

    # Load data
    desi = stream_funcs.load_fits_columns(desi_path, desired_columns, [1, 4])
    dist = stream_funcs.load_fits_columns(distance_path, dist_columns)
    decals = stream_funcs.load_fits_columns(decals_path, decal_columns)
    data = table.hstack([desi, dist, decals])

    # Fix duplicate TARGETID
    if 'TARGETID_2' in data.colnames:
        if np.all(data['TARGETID_1'] == data['TARGETID_2']):
            data.remove_columns(['TARGETID_2'])
            data.rename_column('TARGETID_1', 'TARGETID')

    # Drop NaNs
    drop_columns = desired_columns + decal_columns
    data = stream_funcs.dropna_Table(data, columns=drop_columns)

    # Return boolean selection mask
    return (data['RVS_WARN'] == 0) & \
           (data['RR_SPECTYPE'] == 'STAR') & \
           data['PRIMARY'] & \
           (data['VRAD_ERR'] < 10) & \
           (data['FEH_ERR'] < 0.5)

def import_mcmc_results(stream_run_directory):
    mcmc_dict = np.load(stream_run_directory + '/mcmc_dict.npy', allow_pickle=True).item()
    nested_dict = np.load(stream_run_directory + '/nested_dict.npy', allow_pickle=True).item()
    spline_points_dict = np.load(stream_run_directory + '/spline_points_dict.npy', allow_pickle=True).item()
    return mcmc_dict, nested_dict, spline_points_dict

def mag_select(data, mag_min, mag_max, mag_col='rmag0'):
    return (data[mag_col] < mag_min) & (data[mag_col] > mag_max)

def isochrone_cut(color_indx_wiggle = 0.10, isochrone_path= '/home/jupyter-nassermoha/raid_nassermoha/data/dotter/iso_a13.5_z0.00010.dat', 
                  desi_data=[], desi_distance=18e3, withAss=True):
    dotter_mp = np.loadtxt(isochrone_path)
    # Obtain the M_g and M_r color band data
    dotter_g_mp = dotter_mp[:,6]
    dotter_r_mp = dotter_mp[:,7]

    desi_ebv = np.array(desi_data['EBV'].values)
    desi_g_flux, desi_r_flux = np.array(desi_data['FLUX_G'].values), np.array(desi_data['FLUX_R'].values)

    # Custom function to calculate the color index (g-r), absolute R magnitude, and apparent r magnitude after EBV correction
    desi_colour_index, desi_abs_mag, desi_r_mag = stream_funcs.get_colour_index_and_abs_mag(desi_ebv, desi_g_flux, desi_r_flux, desi_distance)
    # Fit a line to the isochrone data. To use scipy's interpolate properly, the absolute magnitude values must be increasing, which is why we sort the values first

    g_r_color_dif = dotter_g_mp - dotter_r_mp
    sorted_indices = np.argsort(dotter_r_mp)
    sorted_dotter_r_mp = dotter_r_mp[sorted_indices]
    g_r_color_dif = g_r_color_dif[sorted_indices]

    # Fit for the isochrone line
    isochrone_fit = sp.interpolate.UnivariateSpline(sorted_dotter_r_mp, g_r_color_dif, s=0)

    # Cut around the isochrone by the amount specified in color_indx_wiggle
    isochrone_cut = stream_funcs.betw(desi_colour_index, isochrone_fit(desi_abs_mag), color_indx_wiggle) 

    bhb_color_wiggle = 0.4
    bhb_abs_mag_wiggle = 0.1

    # build the BHB using empirical data from M92
    dm_m92_harris = 14.59 #dm of M92
    m92ebv = 0.023
    m92ag = m92ebv * 3.184
    m92ar = m92ebv * 2.130
    m92_hb_r = np.array([17.3, 15.8, 15.38, 15.1, 15.05])
    m92_hb_col = np.array([-0.39, -0.3, -0.2, -0.0, 0.1])
    m92_hb_g = m92_hb_r + m92_hb_col
    des_m92_hb_g = m92_hb_g - 0.104 * (m92_hb_g - m92_hb_r) + 0.01
    des_m92_hb_r = m92_hb_r - 0.102 * (m92_hb_g - m92_hb_r) + 0.02
    des_m92_hb_g = des_m92_hb_g - m92ag - dm_m92_harris
    des_m92_hb_r = des_m92_hb_r - m92ar - dm_m92_harris

    dm = 5 * np.log10(desi_distance) - 5


    bhb_cut = stream_funcs.isochrone_btw(desi_colour_index, desi_abs_mag, bhb_abs_mag_wiggle, bhb_color_wiggle, des_m92_hb_g - des_m92_hb_r, des_m92_hb_r)

    if withAss:
        return isochrone_cut | bhb_cut
    else:
        print('Not applying BHB cut')
        return isochrone_cut

class DataHandler:
    def __init__(self, frame, spline_points_dict, nested_dict):
        self.frame = frame
        self.spline_points_dict = spline_points_dict
        self.nested_dict = nested_dict
        self.data = None
        self.mask = None
    
    def apply_box_cut(self, pad=np.array([30. , -2. ,  0.5,  0.5, 10. ]), upper_bound_feh=True, MaskList=['VGSR', 'FEH', 'PMRA', 'PMDEC', 'phi2']):
        self.touch_masks = {}  # initialize before the loop

        if self.data is not None:
            for i, MaskReturn in enumerate(MaskList):
                    k=i
                    if i==4:
                        k=3
                    touch_mask = get_touch_mask(
                        data=self.data, 
                        MaskReturn=MaskReturn,
                        pad=pad[i],
                        nested_list_meds=self.nested_dict['meds'], 
                        phi1_spline_points=self.spline_points_dict['phi1_spline_points'],
                        spline_k=self.spline_points_dict['spline_k'], 
                        upper_bound=upper_bound_feh,
                        meds_ind=[1, 3, 5, 6][k]
                    )
                    self.touch_masks[MaskReturn] = touch_mask

    def apply_plx_cut(self, min_dist):
        if self.data is not None:
            plx = self.data['PARALLAX']
            plx_err = self.data['PARALLAX_ERROR']
            self.plx_mask = plx_mask(min_dist, plx, plx_err)
        
    def apply_phi1_mask(self, phi1_min=-20, phi1_max=50):
        if self.data is not None:
            phi1 = self.data['phi1']
            self.phi1_mask = (phi1 > phi1_min) & (phi1 < phi1_max)

    def notmask(self,MaskList=['VGSR', 'FEH', 'PMRA', 'PMDEC', 'phi2']):
        """
        Returns a combined mask (logical AND) of all masks in handler except the ones in exclude_keys.

        Parameters:
        - handler: object with touch_masks dict and other masks as attributes
        - exclude_keys: list of mask names to exclude, e.g., ["VGSR", "FEH"]
        """
        # Assuming self.touch_masks, self.plx_mask, self.phi1_mask are already defined
        # and MaskList is an iterable of iterables (e.g., list of lists of strings)

        self.not_masks = {}
        for i, exclude_keys_list in enumerate(MaskList): # Renamed to avoid confusion
            masks = None

            # Mapping of keys to their source
            key_to_mask = {
                "VGSR": self.touch_masks['VGSR'],
                "FEH": self.touch_masks['FEH'],
                "PMRA": self.touch_masks['PMRA'],
                "PMDEC": self.touch_masks['PMDEC'],
                "phi2": self.touch_masks['phi2'],
                "plx": self.plx_mask,
                "phi1": self.phi1_mask
            }

            # Ensure exclude_keys_list is in a consistent format for "not in" check, e.g., a set
            # If exclude_keys_list could be a single string, you'd need to handle that:
            if isinstance(exclude_keys_list, str):
                current_exclude_set = {exclude_keys_list}
                # For the key, we'll still use a tuple of one item
                dict_key_tuple = (exclude_keys_list,)
            else:
                # Convert to set for efficient "not in" lookup
                current_exclude_set = set(exclude_keys_list)
                # Create a sorted tuple for the dictionary key for consistency
                # (e.g., ['FEH', 'PMRA'] and ['PMRA', 'FEH'] become the same key)
                dict_key_tuple = tuple(sorted(list(current_exclude_set)))


            for key_name in key_to_mask.keys():
                if key_name not in current_exclude_set:
                    if masks is None:
                        masks = key_to_mask[key_name]
                    else:
                        masks = masks & key_to_mask[key_name]

            if masks is None:
                # This means current_exclude_set contained all keys from key_to_mask
                raise ValueError(f"All masks were excluded for exclusion set: {current_exclude_set}, no mask to combine.")

            self.not_masks[dict_key_tuple] = masks

    def all_box_cuts_DESI(self, withIso=True, withAss=True):
        """
        Plotting function for applying masks excluding the current panel. Used for box_plot figures
        """
        if hasattr(self, 'touch_masks') is True:
            sel_vgsr = self.touch_masks['VGSR']
            sel_feh = self.touch_masks['FEH']
            sel_pmra = self.touch_masks['PMRA']
            sel_pmdec = self.touch_masks['PMDEC']
            sel_phi2 = self.touch_masks['phi2']
        else:
            # the masks should all be true
            sel_vgsr = np.ones(len(self.data), dtype=bool)
            sel_feh = np.ones(len(self.data), dtype=bool)
            sel_pmra = np.ones(len(self.data), dtype=bool)
            sel_pmdec = np.ones(len(self.data), dtype=bool)
            sel_phi2 = np.ones(len(self.data), dtype=bool)
        sel_plx = self.plx_mask
        sel_phi1 = self.phi1_mask
        if withIso:
            sel_iso = isochrone_cut(color_indx_wiggle=0.10, isochrone_path='/home/jupyter-nassermoha/raid_nassermoha/data/dotter/iso_a13.5_z0.00010.dat',
                    desi_data=self.data, desi_distance=18e3, withAss=withAss)
            sel = sel_vgsr & sel_feh & sel_pmra & sel_pmdec & sel_phi2 & sel_plx & sel_phi1 & sel_iso
        else:
            sel = sel_vgsr & sel_feh & sel_pmra & sel_pmdec & sel_phi2 & sel_plx & sel_phi1

        self.sel = sel

class GaiaDeCALSHandler:
    def __init__(self, frame, spline_points_dict, nested_dict, GaiaDeCALS_data=None, stream_data=None, isochrone_path=None, min_dist=None):
        self.frame = frame
        self.spline_points_dict = spline_points_dict
        self.spline_k = spline_points_dict['spline_k']
        self.nested_dict = nested_dict
        self.data = GaiaDeCALS_data
        self.stream_data = stream_data
        self.isochrone_path = isochrone_path
        self.min_dist = min_dist

        print(f"Duplicates: {sum(self.data.duplicated(subset=['source_id']))}")
        self.data = self.data[self.data['release'] != 9011]
        print(f"Duplicates remaining: {sum(self.data.duplicated(subset=['source_id']))}")

        # Dealing with duplicates, keeping the instance with the smallest delta_mag
        self.data['delta_mag'] = np.abs(self.data['rmag']-self.data['phot_g_mean_mag'])
        self.data.sort_values('delta_mag', inplace=True)
        self.data = self.data[~self.data.duplicated(subset=['source_id'], keep='first')]
        # check if there are any duplicates remaining
        print(f"Duplicates remaining: {sum(self.data.duplicated(subset=['source_id']))}")

        print('Converting to phi1, phi2...')
        self.data['phi1'], self.data['phi2'] = stream_funcs.ra_dec_to_phi1_phi2(self.frame, np.array(self.data['ra'])*u.deg, np.array(self.data['dec'])*u.deg)

        print('showing GaiaDeCALS data in stream frame...')
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.scatter(self.data['phi1'], self.data['phi2'], s=1, color='k', alpha=0.5, label='GaiaDeCALS')
        ax.scatter(self.stream_data['phi1'], self.stream_data['phi2'], s=5, color='orange', alpha=1, label='Stream')
        ax.set_xlabel(r'$\phi_1$ (deg)')
        ax.set_ylabel(r'$\phi_2$ (deg)')
        #show legend
        ax.legend(loc='upper right')
        plot_form(ax)

    def apply_box_cut(
        self,
        pad=np.array([0.3, 0.3, 10]),
        phi1_spline_points=None,
        nested_list_meds=None,
        blind_panels=['pmra', 'pmdec', 'phi2'],
        blind_meds_ind=[5, 6],
        spline_k=None
    ):
        """ Apply box cuts to the GaiaDeCALS data based on the given pad values."""

        # Assign defaults from self if not provided
        if phi1_spline_points is None:
            phi1_spline_points = self.spline_points_dict['phi1_spline_points']
        if nested_list_meds is None:
            nested_list_meds = self.nested_dict['meds']
        if spline_k is None:
            spline_k = self.spline_k

        masks_dict = {}
        for i, panel in enumerate(blind_panels):
            if panel == 'phi2':
                reference_value = 0
            else:
                reference_value = reference_value = apply_spline(self.data['phi1'], phi1_spline_points, nested_list_meds[blind_meds_ind[i]], spline_k)
            mask = (np.abs(self.data[panel] - reference_value) < pad[i])
            masks_dict[panel] = mask
        
        sel_pmra, sel_pmdec, sel_phi2 = (masks_dict['pmra'], masks_dict['pmdec'], masks_dict['phi2'])
        sel_plx = plx_mask(self.min_dist/1000, self.data['parallax'], self.data['parallax_error'])

        x_arr = np.linspace(-15, 45, 1000)
        fig, ax = plt.subplots(3, 1, figsize=(8, 8))
        ax[0].plot(x_arr, np.zeros_like(x_arr), color='cyan', ls='--', zorder=0)
        ax[1].plot(x_arr, apply_spline(x_arr, phi1_spline_points, nested_list_meds[self.pmra_idx], k=spline_k), color='cyan', ls='--', zorder=0)
        ax[2].plot(x_arr, apply_spline(x_arr, phi1_spline_points, nested_list_meds[self.pmdec_idx], k=spline_k), color='cyan', ls='--', zorder=0)

        ax[1].plot(x_arr, apply_spline(x_arr, phi1_spline_points, nested_list_meds[self.pmra_idx], k=spline_k) - pad[0], color='blue', lw=0.5)
        ax[1].plot(x_arr, apply_spline(x_arr, phi1_spline_points, nested_list_meds[self.pmra_idx], k=spline_k) + pad[0], color='blue', lw=0.5)
        ax[2].plot(x_arr, apply_spline(x_arr, phi1_spline_points, nested_list_meds[self.pmdec_idx], k=spline_k) - pad[1], color='blue', lw=0.5)
        ax[2].plot(x_arr, apply_spline(x_arr, phi1_spline_points, nested_list_meds[self.pmdec_idx], k=spline_k) + pad[1], color='blue', lw=0.5)

        ax[0].scatter(self.data['phi1'][sel_pmra & sel_pmdec & sel_plx & sel_phi2], self.data['phi2'][sel_pmra & sel_pmdec & sel_plx & sel_phi2], s=0.5, alpha=0.1, c='b')
        ax[1].scatter(self.data['phi1'][sel_phi2 & sel_pmdec & sel_plx & sel_pmra], self.data['pmra'][sel_phi2 & sel_pmdec & sel_plx & sel_pmra], s=0.5, alpha=0.1, c='b')
        ax[2].scatter(self.data['phi1'][sel_phi2 & sel_pmra & sel_plx & sel_pmdec], self.data['pmdec'][sel_phi2 & sel_pmra & sel_plx & sel_pmdec], s=0.5, alpha=0.1, c='b')

        ax[0].scatter(self.data['phi1'][sel_pmra & sel_pmdec & sel_plx], self.data['phi2'][sel_pmra & sel_pmdec & sel_plx], s=0.1, alpha=0.1, c='0.5')
        ax[1].scatter(self.data['phi1'][sel_phi2 & sel_pmdec & sel_plx], self.data['pmra'][sel_phi2 & sel_pmdec & sel_plx], s=0.1, alpha=0.1, c='0.5')
        ax[2].scatter(self.data['phi1'][sel_phi2 & sel_pmra & sel_plx], self.data['pmdec'][sel_phi2 & sel_pmra & sel_plx], s=0.1, alpha=0.1, c='0.5')

        # plot stream data as orange
        if self.stream_data is not None:
            ax[0].scatter(self.stream_data['phi1'], self.stream_data['phi2'], s=1, color='orange', alpha=1, label='Stream')
            ax[1].scatter(self.stream_data['phi1'], self.stream_data['PMRA'], s=1, color='orange', alpha=1)
            ax[2].scatter(self.stream_data['phi1'], self.stream_data['PMDEC'], s=1, color='orange', alpha=1)

        ax[0].set_ylabel(r'$\phi_2$ (deg)')
        ax[1].set_ylabel(r'$\mu_{\alpha}$ (mas/yr)')
        ax[2].set_ylabel(r'$\mu_{\delta}$ (mas/yr)')

        ax[0].set_ylim(pad[-1] * -1.5, pad[-1] * 1.5)
        ax[1].set_ylim(apply_spline(x_arr, phi1_spline_points, nested_list_meds[self.pmra_idx], k=spline_k).min() - pad[0] * 1.4,
                    apply_spline(x_arr, phi1_spline_points, nested_list_meds[self.pmra_idx], k=spline_k).max() + pad[0] * 1.4)
        ax[2].set_ylim(apply_spline(x_arr, phi1_spline_points, nested_list_meds[self.pmdec_idx], k=spline_k).min() - pad[1] * 1.4,
                    apply_spline(x_arr, phi1_spline_points, nested_list_meds[self.pmdec_idx], k=spline_k).max() + pad[1] * 1.4)


        for a in ax:
            plot_form(a)

        print(f'Applied cuts: {pad} for panels: {blind_panels}, with a parallax cut')
        self.sel_pmra = sel_pmra
        self.sel_pmdec = sel_pmdec
        self.sel_plx = sel_plx
        self.sel_phi2 = sel_phi2
        self.sel_box = sel_pmra & sel_pmdec & sel_plx & sel_phi2

    def apply_iso_cut(self, color_indx_wiggle=None):

        sel_pmra = self.sel_pmra
        sel_pmdec = self.sel_pmdec
        sel_phi2 = self.sel_phi2
        sel_plx = self.sel_plx
        stream_data = self.stream_data
        desi_colour_index, desi_abs_mag, desi_r_mag = stream_funcs.get_colour_index_and_abs_mag(stream_data['EBV'], stream_data['FLUX_G'], stream_data['FLUX_R'], self.min_dist)

        color_indx_wiggle = color_indx_wiggle if color_indx_wiggle is not None else 0.05

        dotter_mp = np.loadtxt(self.isochrone_path)
        self.dotter_mp = dotter_mp
        dotter_g_mp = dotter_mp[:,6] # Absolute magnitude gband, M_g
        dotter_r_mp = dotter_mp[:,7] # Absolute magnitude rband, M_r
        dotter_z_mp = dotter_mp[:,9] # Absolute magnitude zband, M_z

        g_r_color_dif = dotter_g_mp - dotter_r_mp # color
        sorted_indices = np.argsort(dotter_r_mp)  # sorting to go from most green to most red
        sorted_dotter_r_mp = dotter_r_mp[sorted_indices] 
        g_r_color_dif = g_r_color_dif[sorted_indices] 

        isochrone_fit = sp.interpolate.UnivariateSpline(sorted_dotter_r_mp, g_r_color_dif, s=0) # function of colour as a function of absolute magnitude
        gaia_colour_index, gaia_abs_mag, gaia_r_mag = stream_funcs.get_colour_index_and_abs_mag(self.data['ebv'], self.data['flux_g'], self.data['flux_r'], self.min_dist)
        sel_iso = abs(self.data['gmag0']-self.data['rmag0']-apply_spline(self.data['rmag0']-d2dm(self.min_dist/1000),dotter_r_mp[::-1], dotter_g_mp[::-1]-dotter_r_mp[::-1]-0.01, k=1)) < color_indx_wiggle

        mag_min = 20.5
        mag_max = 16.0
        sel_mag = mag_select(self.data, mag_min=mag_min, mag_max=mag_max, mag_col='rmag0')
        "CHANGE ABOVE TO GET DIFFERENT MAG CUTS, NOT GONNA MAKE THIS PART USER FRIENDLY SORRY"

        self.sel_iso = sel_iso 
        self.sel_mag = sel_mag

        fig, ax = plt.subplots(1, 1, figsize=(4, 6))

        sel2 = self.sel_mag & sel_pmra & sel_pmdec & sel_phi2 & sel_plx & sel_iso
        ax.scatter(gaia_colour_index[self.sel_pmra & self.sel_mag & self.sel_pmdec & self.sel_phi2 & self.sel_plx & ~self.sel_iso], gaia_abs_mag[self.sel_pmra & self.sel_mag & self.sel_pmdec & self.sel_phi2 & self.sel_plx & ~self.sel_iso], color='0.2', s=0.5, alpha=0.05)
        ax.scatter(gaia_colour_index[self.sel_pmra & self.sel_mag & self.sel_pmdec & self.sel_phi2 & self.sel_plx & self.sel_iso], gaia_abs_mag[self.sel_pmra & self.sel_mag & self.sel_pmdec & self.sel_phi2 & self.sel_plx & self.sel_iso], color='blue', s=1, alpha=0.1)
        ax.scatter(desi_colour_index, desi_abs_mag, color='orange', s=10, alpha=1, marker='s', label=r'Stream Stars')
        ax.plot(isochrone_fit(sorted_dotter_r_mp), sorted_dotter_r_mp, color='red', lw=2, label='Isochrone Fit')
        ax.plot(isochrone_fit(sorted_dotter_r_mp)+color_indx_wiggle, sorted_dotter_r_mp, color='red', lw=1, ls='-.')
        ax.plot(isochrone_fit(sorted_dotter_r_mp)-color_indx_wiggle, sorted_dotter_r_mp, color='red', lw=1, ls='-.')
        ax.axhline(mag_max-5*np.log10(18e3)+5, ls='dotted', c='k', label=rf'$r={mag_max}$')
        ax.axhline(mag_min-5*np.log10(18e3)+5, ls='dotted', c='k', label=rf'$r={mag_min}$')

        ax.invert_yaxis()
        ax.set_ylim(5, -3)
        ax.set_xlim(-0.5, 1)
        ax.set_xlabel('Colour Index (g - r)')
        ax.set_ylabel('Absolute Magnitude (M_r)')
        ax.set_title(r'Gaia x DECaLS Data: Isochrone Cut from $r \in$'+ f'({mag_min}, {mag_max})')
        ax.legend()
        plot_form(ax)

        self.sel_all = sel_pmra & sel_pmdec & sel_phi2 & sel_plx & sel_iso & sel_mag
    
    def apply_phot_metallicity(self):
        from scipy.interpolate import interp1d
        mpsel = self.data['rmag0']-self.data['zmag0'] - (0.24/0.25)*(self.data['gmag0']-self.data['rmag0']-0.5) - 0.24 > 0 # Empirical, from Ting
        sel_all_old = self.sel_all
        self.sel_all = sel_all_old & mpsel

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        ax.scatter(self.data['gmag0'][sel_all_old]-self.data['rmag0'][sel_all_old], self.data['rmag0'][sel_all_old]-self.data['zmag0'][sel_all_old], marker='.', s=10, c='0.2')
        ax.scatter(self.data['gmag0'][self.sel_all]-self.data['rmag0'][self.sel_all], self.data['rmag0'][self.sel_all]-self.data['zmag0'][self.sel_all], marker='.', s=10, c='blue')

        #ax.scatter(self.stream_data['gmag0']-self.stream_data['rmag0'], self.stream_data['rmag0']-self.stream_data['zmag0'], color='tab:orange', marker='s')
        #ax.scatter(stream_data_joined['gmag0']-stream_data_joined['rmag0'], stream_data_joined['rmag0']-stream_data_joined['zmag0'], color='tab:orange', marker='x')

        ax.set_xlabel('g-r',fontsize=15)
        ax.set_ylabel('r-z',fontsize=15)
        ax.set_aspect('equal')
        #ax.plot(dotter_g_mp[s:r] - dotter_r_mp[s:r], dotter_r_mp[s:r] - dotter_z_mp[s:r], color='0.2', lw=1,label='Isochrone Fit')
        #ax.plot(dotter_g_mp[s:r] - dotter_r_mp[s:r], dotter_r_mp[s:r] - dotter_z_mp[s:r], color='0.5', lw=1,label='Isochrone Fit')

        #ax.plot(dotter_g_mp - dotter_r_mp, dotter_r_mp - dotter_z_mp, color='0.2', lw=1, ls = 'dotted', label='Isochrone Fit')

        ax.set_xlim(0.1,0.8)
        ax.set_ylim(-0.1,0.5)

        x = np.linspace(0.1,0.8,100)
        y = (0.24/0.25)*(x-0.5) + 0.24

        ax.plot(x,y, c='red')

        plot_form(ax)
def dm_to_distance(dm, dmerr=None, dmerr_plus=None, dmerr_minus=None, to_kpc=False):
    """
    Convert distance modulus (dm) to linear distance, returning asymmetric
    or symmetric uncertainties as appropriate.

    Parameters
    ----------
    dm : float or array-like
        Distance modulus.
    dmerr : float or array-like, optional
        1-σ *symmetric* uncertainty on dm.  Ignored if dmerr_plus/minus are given.
    dmerr_plus : float or array-like, optional
        Positive (upper) uncertainty on dm.
    dmerr_minus : float or array-like, optional
        Negative (lower) uncertainty on dm.
    to_kpc : bool, default False
        If True, distances are returned in kiloparsecs instead of parsecs.

    Returns
    -------
    d : float or ndarray
        Central distance (pc or kpc).
    derr_plus : float or ndarray
        Upper uncertainty on distance (same units as d).
    derr_minus : float or ndarray
        Lower uncertainty on distance (same units as d).
        For symmetric-error input, derr_plus == derr_minus.
    """
    # central distance
    d = 10.0**((np.asarray(dm) + 5.0) / 5.0)

    # choose error treatment
    if dmerr_plus is not None and dmerr_minus is not None:
        # asymmetric case
        d_hi = 10.0**(((dm + dmerr_plus) + 5.0) / 5.0)
        d_lo = 10.0**(((dm - dmerr_minus) + 5.0) / 5.0)
        derr_plus  = d_hi - d
        derr_minus = d - d_lo
    elif dmerr is not None:
        # symmetric case – linear propagation
        factor = np.log(10.0) / 5.0           # ∂d/∂dm divided by d
        derr_plus = derr_minus = d * factor * dmerr
    else:
        raise ValueError("Provide either dmerr (symmetric) or both dmerr_plus and dmerr_minus.")

    if to_kpc:
        d          = d / 1e3
        derr_plus  = derr_plus  / 1e3
        derr_minus = derr_minus / 1e3

    return d, [derr_minus, derr_plus]
def get_touch_mask(data, MaskReturn=[], pad=[], nested_list_meds=[], phi1_spline_points=[], meds_ind = 1, spline_k=1, upper_bound=True):
    """
    Create a mask for called value
    """
    mask = None
    if MaskReturn=='FEH':
        if upper_bound:
            mask = data[MaskReturn] < pad
        else:
            reference_value = nested_list_meds[3]
            mask = (np.abs(data[MaskReturn] - reference_value) < pad)
    else:
        if MaskReturn == 'phi2':
            reference_value = 0
        else:
            reference_value = apply_spline(data['phi1'], phi1_spline_points, nested_list_meds[meds_ind], spline_k)
        mask = (np.abs(data[MaskReturn] - reference_value) < pad)
    return mask

    
def with_plotting_box_panels(fig, ax, handler, label='MS+RG', pad=[], highlight=None):
    """
    Plotting function for applying masks excluding the current panel. Used for box_plot figures
    """
    sel_vgsr = handler.touch_masks['VGSR']
    sel_feh = handler.touch_masks['FEH']
    sel_pmra = handler.touch_masks['PMRA']
    sel_pmdec = handler.touch_masks['PMDEC']
    sel_phi2 = handler.touch_masks['phi2']
    sel_plx = handler.plx_mask
    sel_phi1 = handler.phi1_mask
    sel = sel_vgsr & sel_feh & sel_pmra & sel_pmdec & sel_phi2 & sel_plx & sel_phi1
    sel_nophi2 = sel_vgsr & sel_feh & sel_pmra & sel_pmdec & sel_plx & sel_phi1
    sel_novgsr = sel_feh & sel_pmra & sel_pmdec & sel_plx & sel_phi1 & sel_phi2
    sel_nopmra = sel_vgsr & sel_feh & sel_pmdec & sel_plx & sel_phi1 & sel_phi2
    sel_nopmdec = sel_vgsr & sel_feh & sel_pmra & sel_plx & sel_phi1 & sel_phi2
    sel_nofeh = sel_vgsr & sel_pmra & sel_pmdec & sel_plx & sel_phi1 & sel_phi2
    x_arr = np.arange(-10, 40, 0.1)
    if label == 'MS+RG':
        marker = 'o'
        color = 'k'
        sel_iso = isochrone_cut(color_indx_wiggle=0.10, isochrone_path='/home/jupyter-nassermoha/raid_nassermoha/data/dotter/iso_a13.5_z0.00010.dat',
                  desi_data=handler.data, desi_distance=18e3, withAss=False)
        sel = sel & sel_iso
        sel_nophi2 = sel_nophi2 & sel_iso
        sel_novgsr = sel_novgsr & sel_iso
        sel_nopmra = sel_nopmra & sel_iso
        sel_nopmdec = sel_nopmdec & sel_iso
        sel_nofeh = sel_nofeh & sel_iso
        handler.sel = sel
    elif label == 'BHB':
        marker = '^'
        color = 'b'
        handler.sel = sel
    elif label == 'RRL':
        marker = 'v'
        color = 'magenta'
        handler.sel = sel

    ax[0].scatter(handler.data['phi1'][sel_nophi2], handler.data['phi2'][sel_nophi2], edgecolor=color, facecolor='white', s=30, zorder=1, label=label, marker=marker)
    ax[0].axhline(0, color='k', lw=1, zorder=2)
    ax[0].axhline(pad[4], color='k', lw=0.5, ls='-.', zorder=0)
    ax[0].axhline(-pad[4], color='k', lw=0.5, ls='-.', zorder=0)

    ax[1].scatter(handler.data['phi1'][sel_novgsr], handler.data['VGSR'][sel_novgsr], edgecolor=color, facecolor='white', s=30, zorder=1, label=label, marker=marker)
    ax[1].plot(x_arr, apply_spline(x_arr, handler.spline_points_dict['phi1_spline_points'], handler.nested_dict['meds'][self.vgsr_idx], handler.spline_points_dict['spline_k']), color='k', lw=1, zorder=0)
    ax[1].plot(x_arr, apply_spline(x_arr, handler.spline_points_dict['phi1_spline_points'], handler.nested_dict['meds'][self.vgsr_idx], handler.spline_points_dict['spline_k'])+pad[0], color='k', lw=0.5, ls='-.', zorder=0)
    ax[1].plot(x_arr, apply_spline(x_arr, handler.spline_points_dict['phi1_spline_points'], handler.nested_dict['meds'][self.vgsr_idx], handler.spline_points_dict['spline_k'])-pad[0], color='k', lw=0.5, ls='-.', zorder=0)
    ax[2].scatter(handler.data['phi1'][sel_novgsr],handler.data['VGSR'][sel_novgsr]- apply_spline(handler.data['phi1'][sel_novgsr], handler.spline_points_dict['phi1_spline_points'], handler.nested_dict['meds'][self.vgsr_idx], handler.spline_points_dict['spline_k']), edgecolor=color, facecolor='white', s=30, zorder=1, label=label, marker=marker)
    ax[2].axhline(0, color='k', lw=1, zorder=2)
    ax[2].axhline(pad[0], color='k', lw=0.5, ls='-.', zorder=0)
    ax[2].axhline(-pad[0], color='k', lw=0.5, ls='-.', zorder=0)

    ax[3].scatter(handler.data['phi1'][sel_nopmra], handler.data['PMRA'][sel_nopmra], edgecolor=color, facecolor='white', s=30, zorder=1, label=label, marker=marker)
    ax[3].plot(x_arr, apply_spline(x_arr, handler.spline_points_dict['phi1_spline_points'], handler.nested_dict['meds'][self.pmra_idx], handler.spline_points_dict['spline_k']), color='k', lw=1, zorder=0)
    ax[3].plot(x_arr, apply_spline(x_arr, handler.spline_points_dict['phi1_spline_points'], handler.nested_dict['meds'][self.pmra_idx], handler.spline_points_dict['spline_k'])+pad[2], color='k', lw=0.5, ls='-.', zorder=0)
    ax[3].plot(x_arr, apply_spline(x_arr, handler.spline_points_dict['phi1_spline_points'], handler.nested_dict['meds'][self.pmra_idx], handler.spline_points_dict['spline_k'])-pad[2], color='k', lw=0.5, ls='-.', zorder=0)
    ax[4].scatter(handler.data['phi1'][sel_nopmra], handler.data['PMRA'][sel_nopmra] - apply_spline(handler.data['phi1'][sel_nopmra], handler.spline_points_dict['phi1_spline_points'], handler.nested_dict['meds'][self.pmra_idx], handler.spline_points_dict['spline_k']), edgecolor=color, facecolor='white', s=30, zorder=1, label=label, marker=marker)
    ax[4].axhline(0, color='k', lw=1, zorder=2)
    ax[4].axhline(pad[2], color='k', lw=0.5, ls='-.', zorder=0)
    ax[4].axhline(-pad[2], color='k', lw=0.5, ls='-.', zorder=0)

    ax[5].scatter(handler.data['phi1'][sel_nopmdec], handler.data['PMDEC'][sel_nopmdec], edgecolor=color, facecolor='white', s=30, zorder=1, label=label, marker=marker)
    ax[5].plot(x_arr, apply_spline(x_arr, handler.spline_points_dict['phi1_spline_points'], handler.nested_dict['meds'][self.pmdec_idx], handler.spline_points_dict['spline_k']), color='k', lw=1, zorder=0)
    ax[5].plot(x_arr, apply_spline(x_arr, handler.spline_points_dict['phi1_spline_points'], handler.nested_dict['meds'][self.pmdec_idx], handler.spline_points_dict['spline_k'])+pad[3], color='k', lw=0.5, ls='-.', zorder=0)
    ax[5].plot(x_arr, apply_spline(x_arr, handler.spline_points_dict['phi1_spline_points'], handler.nested_dict['meds'][self.pmdec_idx], handler.spline_points_dict['spline_k'])-pad[3], color='k', lw=0.5, ls='-.', zorder=0)
    ax[6].scatter(handler.data['phi1'][sel_nopmdec], handler.data['PMDEC'][sel_nopmdec] - apply_spline(handler.data['phi1'][sel_nopmdec], handler.spline_points_dict['phi1_spline_points'], handler.nested_dict['meds'][self.pmdec_idx], handler.spline_points_dict['spline_k']), edgecolor=color, facecolor='white', s=30, zorder=1, label=label, marker=marker)
    ax[6].axhline(0, color='k', lw=1, zorder=2)
    ax[6].axhline(pad[3], color='k', lw=0.5, ls='-.', zorder=0)
    ax[6].axhline(-pad[3], color='k', lw=0.5, ls='-.', zorder=0)

    ax[7].scatter(handler.data['phi1'][sel_nofeh], handler.data['FEH'][sel_nofeh], edgecolor=color, facecolor='white', s=30, zorder=1, label=label, marker=marker)
    ax[7].axhline(handler.nested_dict['meds'][3], color='k', lw=1, zorder=0)
    ax[7].axhline(pad[1], color='k', lw=0.5, ls='-.', zorder=0)

    if highlight is not None:
        ax[0].scatter(highlight['phi1'], highlight['phi2'], s=80, c='m', alpha=0.5, label='highlight')
        ax[1].scatter(highlight['phi1'], highlight['VGSR'], s=80, c='m', alpha=0.5, label='highlight')
        ax[2].scatter(highlight['phi1'], highlight['VGSR'] - apply_spline(highlight['phi1'], handler.spline_points_dict['phi1_spline_points'], handler.nested_dict['meds'][self.vgsr_idx], handler.spline_points_dict['spline_k']), s=80, c='m', alpha=0.5, label='highlight')
        ax[3].scatter(highlight['phi1'], highlight['PMRA'], s=80, c='m', alpha=0.5, label='highlight')
        ax[4].scatter(highlight['phi1'], highlight['PMRA'] - apply_spline(highlight['phi1'], handler.spline_points_dict['phi1_spline_points'], handler.nested_dict['meds'][4], handler.spline_points_dict['spline_k']), s=80, c='m', alpha=0.5, label='highlight')
        ax[5].scatter(highlight['phi1'], highlight['PMDEC'], s=80, c='m', alpha=0.5, label='highlight')
        ax[6].scatter(highlight['phi1'], highlight['PMDEC'] - apply_spline(highlight['phi1'], handler.spline_points_dict['phi1_spline_points'], handler.nested_dict['meds'][self.pmdec_idx], handler.spline_points_dict['spline_k']), s=80, c='m', alpha=0.5, label='highlight')

        #ax[8].scatter(highlight['phi1'], dh, s=80, c='m', alpha=0.5, label='highlight')
        ax[7].scatter(highlight['phi1'], highlight['FEH'], s=80, c='m', alpha=0.5, label='highlight')

    if label == 'RRL':
        ax[8].scatter(handler.data['phi1'][sel], handler.data['dist_FEH'][sel], edgecolor='magenta', marker='v', s=50, facecolors='none', zorder=2) # Shifted, Assuming 'dist_FEH' is correct distance for RRL
    else:
        ax[8].scatter(handler.data['phi1'][sel], stream_funcs.dist_mod_to_dist(handler.data['dist_mod'][sel]) / 1000,  edgecolor=color, facecolor='white', s=30, zorder=1, marker=marker)

def with_errors(fig, ax, handler, label='MS+RG', pad=[]):
    """
    plotting function for errors for applying masks excluding the current panel. Used for box_plot figures
    """
    #sel_not = handler.notmask()
    sel_vgsr = handler.touch_masks['VGSR']
    sel_feh = handler.touch_masks['FEH']
    sel_pmra = handler.touch_masks['PMRA']
    sel_pmdec = handler.touch_masks['PMDEC']
    sel_phi2 = handler.touch_masks['phi2']
    sel_plx = handler.plx_mask
    sel_phi1 = handler.phi1_mask
    sel = sel_vgsr & sel_feh & sel_pmra & sel_pmdec & sel_phi2 & sel_plx & sel_phi1
    sel_novgsr = sel_feh & sel_pmra & sel_pmdec & sel_plx & sel_phi1 & sel_phi2
    sel_nopmra = sel_vgsr & sel_feh & sel_pmdec & sel_plx & sel_phi1 & sel_phi2
    sel_nopmdec = sel_vgsr & sel_feh & sel_pmra & sel_plx & sel_phi1 & sel_phi2
    if label == 'MS+RG':
        sel_iso = isochrone_cut(color_indx_wiggle=0.10, isochrone_path='/home/jupyter-nassermoha/raid_nassermoha/data/dotter/iso_a13.5_z0.00010.dat',
                    desi_data=handler.data, desi_distance=18e3)
        sel = sel & sel_iso
        sel_novgsr = sel_novgsr & sel_iso
        sel_nopmra = sel_nopmra & sel_iso
        sel_nopmdec = sel_nopmdec & sel_iso
    x_arr = np.arange(-10, 40, 0.1)
    ax[2].errorbar(
        handler.data['phi1'][sel_novgsr],
        handler.data['VGSR'][sel_novgsr] - apply_spline(handler.data['phi1'][sel_novgsr], handler.spline_points_dict['phi1_spline_points'], handler.nested_dict['meds'][self.vgsr_idx], handler.spline_points_dict['spline_k']),
        yerr=handler.data['VGSR_ERR'][sel_novgsr],
        fmt='none', color='k', elinewidth=0.75, capsize=2, label=label, zorder=0
    )

    ax[4].errorbar(
        handler.data['phi1'][sel_nopmra],
        handler.data['PMRA'][sel_nopmra] - apply_spline(handler.data['phi1'][sel_nopmra], handler.spline_points_dict['phi1_spline_points'], handler.nested_dict['meds'][self.pmra_idx], handler.spline_points_dict['spline_k']),
        yerr=handler.data['PMRA_ERROR'][sel_nopmra],
        fmt='none', color='k', elinewidth=0.75, capsize=2, label=label, zorder=0
    )

    ax[6].errorbar(
        handler.data['phi1'][sel_nopmdec],
        handler.data['PMDEC'][sel_nopmdec] - apply_spline(handler.data['phi1'][sel_nopmdec], handler.spline_points_dict['phi1_spline_points'], handler.nested_dict['meds'][self.pmdec_idx], handler.spline_points_dict['spline_k']),
        yerr=handler.data['PMDEC_ERROR'][sel_nopmdec],
        fmt='none', color='k', elinewidth=0.75, capsize=2, label=label, zorder=0
    )


class StreamMembers:
    """
    Class to handle post-mcmc processing of stream
    """
    def __init__(self, withBHB=True, withRRL=True, min_dist=18e3,
                stream_data=[],
                stream_run_directory=None,
                isochrone_path='',
                desi_data=None, fr=None):

        self.frame=fr
        # if isochrone path is not provided, use default
        if not isochrone_path:
            print('Using default isochrone path: ...iso_a13.5_z0.00010.dat')
            isochrone_path = '/home/jupyter-nassermoha/raid_nassermoha/data/dotter/iso_a13.5_z0.00010.dat'
        self.isochrone_path = isochrone_path
        self.stream_data, self.stream_run_directory = stream_data, stream_run_directory
        # go to stream_run_directory and get path to fits file that contains C-19-I21_phi2_spline_all%_mem.fits
        # self.stream_run_directory = os.path.dirname(self.stream_run_directory)
        if self.stream_run_directory:
            import glob
            pattern = os.path.join(self.stream_run_directory, "*all%_mem.fits")
            files = glob.glob(pattern)
            if files:
                all_mems_path = files[0]  # take first match
                self.all_memberships = table.Table.read(all_mems_path)
                self.all_memberships = self.all_memberships.to_pandas()
            else:
                raise FileNotFoundError(f"No file ending with 'all%_mem.fits' found in {self.stream_run_directory}.")
        else:
            print('no stream directory given, moving on...')
        

        print('Loading desi_data...')
        # if desi_data is not an array or a pandas DataFrame
        if not isinstance(desi_data, (np.ndarray, pd.DataFrame)):
            self.desi_data = None
            print('No desi_data provided, will not use DESI data')
        else:
            self.desi_data = desi_data
            print('Desi data loaded')
        if stream_run_directory is not None:
            self.mcmc_dict, self.nested_dict, self.spline_points_dict = import_mcmc_results(self.stream_run_directory)

            # Create data handlers
            self.ms_handler = DataHandler(self.frame, self.spline_points_dict, self.nested_dict)
            self.ms_handler.data = self.desi_data
        
        if not isinstance(desi_data, (np.ndarray, pd.DataFrame)):
            print('Skipping BHB and RRL Data')
        else:
            print('getting BHB and RRL data...')
            if withBHB:
                bhb_path = '/raid/DESI/DESI_value_added/bhb/loa/loa_bhb_250116.fits'
                bhb_data = table.Table.read(bhb_path)
                bhb_data['phi1'], bhb_data['phi2'] = stream_funcs.ra_dec_to_phi1_phi2(self.frame, np.array(bhb_data['TARGET_RA'])*u.deg, np.array(bhb_data['TARGET_DEC'])*u.deg)
                bhb_desi = bhb_data.to_pandas().merge(self.ms_handler.data, how='left', on=['TARGETID'], suffixes=('', '_desi'))
                self.bhb_handler = DataHandler(self.frame, self.spline_points_dict, self.nested_dict)
                self.bhb_handler.data = bhb_desi
            else:
                self.bhb_handler = None
            
            if withRRL:
                rrl_path = '/raid/DESI/DESI_value_added/rrl/loa/DESI_loa_VAC_v0.2.csv'
                rrl_data = pd.read_csv(rrl_path)
                rrl_data['phi1'], rrl_data['phi2'] = stream_funcs.ra_dec_to_phi1_phi2(self.frame, np.array(rrl_data['TARGET_RA'])*u.deg, np.array(rrl_data['TARGET_DEC'])*u.deg)
                rrl_data['PMRA'], rrl_data['PMDEC'] = rrl_data['pmra'], rrl_data['pmdec']
                rrl_data['PMRA_ERROR'], rrl_data['PMDEC_ERROR'] = rrl_data['pmra_error'], rrl_data['pmdec_error']
                rrl_data['VGSR'] = np.array(stream_funcs.vhel_to_vgsr(np.array(rrl_data['TARGET_RA'])*u.deg, np.array(rrl_data['TARGET_DEC'])*u.deg, np.array(rrl_data['v0_mean'])*u.km/u.s).value)
                rrl_data['VGSR_ERR'] = rrl_data['v0_std']
                self.rrl_handler = DataHandler(self.frame, self.spline_points_dict, self.nested_dict)
                self.rrl_handler.data = rrl_data
                self.rrl_handler.data['PARALLAX'] = self.rrl_handler.data['parallax']
                self.rrl_handler.data['PARALLAX_ERROR'] = self.rrl_handler.data['parallax_error']
            else:
                self.rrl_handler = None
        self.min_dist = min_dist

        self.vgsr_idx = 1
        self.pmra_idx = 5
        labels = self.mcmc_dict.get('extended_param_labels', []) if getattr(self, 'mcmc_dict', None) is not None else []
        self.pmdec_idx = 7 if 'lsigpmra' in labels else 6

        print('Done')

    def box_cut(self, pad, with_plot=True, save_fig=False, fig_path=None, **kwargs):
        print('Applying box cuts...')
        highlight = kwargs.get('highlight', None)
        self.pad = pad
        self.ms_handler.apply_box_cut(pad=pad) # now have self.ms_handler.touch_masks dictionary
        self.ms_handler.apply_plx_cut(min_dist=self.min_dist) # now have self.ms_handler.plx_mask
        self.ms_handler.apply_phi1_mask() # now have self.ms_handler.phi1_mask
        self.ms_handler.all_box_cuts_DESI(withAss=False)

        self.bhb_handler.apply_box_cut(pad=pad) if self.bhb_handler else None
        self.bhb_handler.apply_phi1_mask() if self.bhb_handler else None
        self.bhb_handler.apply_plx_cut(min_dist=self.min_dist) if self.bhb_handler else None
        self.bhb_handler.all_box_cuts_DESI(withIso=False) if self.bhb_handler else None

        self.rrl_handler.apply_box_cut() if self.rrl_handler else None
        self.rrl_handler.apply_phi1_mask() if self.rrl_handler else None
        self.rrl_handler.apply_plx_cut(min_dist=self.min_dist) if self.rrl_handler else None
        self.rrl_handler.all_box_cuts_DESI(withIso=False) if self.rrl_handler else None
        # concatenate all box cuts
        self.box_data = pd.concat([self.ms_handler.data[self.ms_handler.sel], self.bhb_handler.data[self.bhb_handler.sel], self.rrl_handler.data[self.rrl_handler.sel]], ignore_index=True)
        print('Box cuts applied. Access total masks using StreamMembers.<handler>.sel')
        if with_plot:
            print('Plotting box cuts...')

            import matplotlib.gridspec as gridspec
            fig = plt.figure(figsize=(12, 17*0.9))
            # Create a GridSpec layout with different row heights
            # Giving less height to delta plots at indices 2, 4, and 6
            heights = [1.5, 1.5, 1, 1.5, 1, 1.5, 1, 1.5, 1.5]  # Half height for delta plots
            gs = gridspec.GridSpec(9, 1, height_ratios=heights)

            # Create axes with the custom layout
            ax = []
            for i in range(9):
                ax.append(fig.add_subplot(gs[i]))
                if i > 0:  # Share x-axis except for the first plot
                    ax[i].sharex(ax[0])
                    ax[i].set_xlim(-10, 40)
            ax[0].set_ylabel(r'$\phi_2$ (deg)', fontsize=14)
            ax[1].set_ylabel(r'$V_{GSR}$ (km/s)', fontsize=14)
            ax[2].set_ylabel(r'$\Delta V_{GSR}$ (km/s)', fontsize=14)
            ax[3].set_ylabel(r'$\mu_{\alpha}$ (mas/yr)', fontsize=14)
            ax[4].set_ylabel(r'$\Delta \mu_{\alpha}$ (mas/yr)', fontsize=14)
            ax[5].set_ylabel(r'$\mu_{\delta}$ (mas/yr)', fontsize=14)
            ax[6].set_ylabel(r'$\Delta \mu_{\delta}$ (mas/yr)', fontsize=14)
            ax[7].set_ylabel(r'[Fe/H]', fontsize=14)
            

            ax[0].set_ylim(-15, 15)
            ax[1].set_ylim(-pad[0]*5, pad[0]*5)
            ax[2].set_ylim(-pad[0]*1.5, pad[0]*1.5)
            ax[3].set_ylim(-2, 4)
            ax[4].set_ylim(-pad[2]*1.5, pad[2]*1.5)
            ax[5].set_ylim(-7, 0)
            ax[6].set_ylim(-pad[3]*1.5, pad[3]*1.5)
            
            with_plotting_box_panels(fig, ax, self.ms_handler, pad=pad, label='MS+RG', highlight=highlight)
            with_errors(fig, ax, self.ms_handler, pad=pad)

            if self.bhb_handler:
                with_plotting_box_panels(fig, ax, self.bhb_handler, label='BHB', pad=pad)
                with_errors(fig, ax, self.bhb_handler, label='BHB', pad=pad)
            if self.rrl_handler:
                with_plotting_box_panels(fig, ax, self.rrl_handler, label='RRL', pad=pad)
                with_errors(fig, ax, self.rrl_handler, label='RRL', pad=pad)



            for i, a in enumerate(ax):
                a.grid(ls='-.', alpha=0.2, zorder=-10)
                a.tick_params(axis='both', which='both', direction='in', top=True, right=True)
                a.spines['top'].set_linewidth(1.5)
                a.spines['right'].set_linewidth(1.5)
                a.spines['left'].set_linewidth(1.5)
                a.spines['bottom'].set_linewidth(1.5)
                a.minorticks_on()
                #turn off x-ticks for all but the last plot
                if i < len(ax) - 1:
                    a.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                else:
                    a.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
            # last panel log y axis
            ax[-1].set_ylabel('Distance (kpc)', fontsize=14)
            ax[-1].set_yscale('log')
            ax[-1].set_xlabel(r'$\phi_1$ (deg)', fontsize=14)
            ax[-1].set_ylim(1, 100)

            # create legend handle
            legend_handles = {
                'MS+RG': plt.Line2D([0], [0], marker='o', color='w', label='MS+RG',
                                    markeredgecolor='k', markerfacecolor='none', markersize=6, linestyle='None'),
                'BHB': plt.Line2D([0], [0], marker='^', color='w', label='BHB',
                                markerfacecolor='none', markeredgecolor='b', markersize=6, linestyle='None'),
                'RRL': plt.Line2D([0], [0], marker='v', color='w', label='RRL',
                                markerfacecolor='none', markeredgecolor='magenta', markersize=6, linestyle='None')
            }
            # Add legend to the first subplot
            ax[0].legend(handles=legend_handles.values(), loc='upper right', fontsize=12, frameon=True, handletextpad=0.5, labelspacing=0.5)
            plt.tight_layout()
            
        
#
            for ax in plt.gcf().get_axes():
                for artist in ax.get_children():
                    artist.set_rasterized(True)
            if save_fig:
                if fig_path is None:
                    fig_path = 'figures_draft/postmcmc_box.pdf'
                plt.savefig(fig_path, bbox_inches='tight', dpi=600)
            plt.show()

    def do_orbit(self, progenitor_RA, theta_init=None, with_plot=True, use_mcmc=True, **kwargs):
        """
        Run the orbit for the stream and plot the results.
        """
        # if orbit_kwargs is None:
        #     orbit_kwargs = {
        #         'fw' : np.linspace(0., 0.2, 2001) * u.Gyr,
        #         'bw' : np.linspace(0, -0.2, 2001) * u.Gyr,
        #         'progenitor_distance': self.min_dist/1000, # in kpc
        #     }
        orbit_kwargs = {
            'fw' :  kwargs.get('fw',  np.linspace(0., 0.2, 2001) * u.Gyr),
            'bw' : kwargs.get('bw',  np.linspace(0., 0.2, 2001) * u.Gyr),
            'progenitor_distance': self.min_dist/1000, # in kpc
        }

        if theta_init is None:
            if orbit_kwargs is None:
                progenitor_dist = self.min_dist / 1000  # in kpc
            else:
                progenitor_dist = orbit_kwargs['progenitor_distance']  # in kpc
            self.stream_data['RA'] = np.array(self.stream_data['TARGET_RA'])
            self.stream_data['DEC'] = np.array(self.stream_data['TARGET_DEC'])
            progenitor_RA = np.mean(self.stream_data['RA'])
            distances = np.abs(self.stream_data['RA'] - progenitor_RA) #finding the stars closest to the progenitor RA

            k = np.argsort(distances)
            guess = {
                    'DEC': np.nanmean(self.stream_data[k]['DEC'][:10]),
                    'PMRA': np.nanmean(self.stream_data[k]['PMRA'][:10]),
                    'PMDEC': np.nanmean(self.stream_data[k]['PMDEC'][:10]),
                    'VRAD': np.nanmean(self.stream_data[k]['VRAD'][:10]),
                    'DIST': progenitor_dist
                }
            lsig_dec_init = np.log10(0.1)  # Example: log10(0.1 deg)
            lsig_pmra_init = np.log10(0.1) # Example: log10(0.1 mas/yr)
            lsig_pmdec_init = np.log10(0.1)# Example: log10(0.1 mas/yr)
            lsig_vrad_init = np.log10(1.0) # Example: log10(1 km/s)

            theta_init = [
                guess['DEC'],
                guess['PMRA'],
                guess['PMDEC'],
                guess['VRAD'],
                guess['DIST'],
                lsig_dec_init,
                lsig_pmra_init,
                lsig_pmdec_init,
                lsig_vrad_init
            ]
        
        self.fw = orbit_kwargs['fw']
        self.bw = orbit_kwargs['bw']

        results_o, orbit = ofuncs.fit_orbit(
            self.stream_data,
            self.frame,
            progenitor_RA,
            orbit_kwargs["fw"],
            orbit_kwargs["bw"],
            theta_init,
            use_position=True, use_mcmc=use_mcmc, nwalkers = kwargs.get('nwalkers', 50), nsteps = kwargs.get('nsteps', 1000)
        )
        self.orbit_ran = True
        if with_plot:
            o_ra, o_dec, o_pmra, o_pmdec, o_vrad, o_dist = ofuncs.orbit_model(results_o.x[0:5], progenitor_RA, orbit_kwargs["fw"], orbit_kwargs["bw"])
            fig, ax = ofuncs.plot_orbit(o_ra, o_dec, self.stream_data['RA'], self.stream_data['DEC'], progenitor_RA)
        self.results_o = results_o
        return results_o, orbit, fig if with_plot else None, ax if with_plot else None
    
    def add_orbit_track(self, ax, results_o, track=''):
        """
        Add the orbit track to the given axis.
        """
        o_ra, o_dec, o_pmra, o_pmdec, o_vrad, o_dist = ofuncs.orbit_model(results_o.x[0:5], np.mean(self.stream_data['RA']), self.fw, self.bw)
        o_phi1, o_phi2 = ofuncs.ra_dec_to_phi1_phi2(self.frame, o_ra*u.deg, o_dec*u.deg)
        o_vgsr = np.array(ofuncs.vhel_to_vgsr(o_ra, o_dec, o_vrad).value)

        importlib.reload(ofuncs)
        ointerps = ofuncs.orbit_interpolations([o_phi1, o_phi2, o_ra, o_dec, o_pmra, o_pmdec, o_vrad, o_vgsr, o_dist])

        orbit_phi1 = np.linspace(np.min(self.stream_data['phi1'] - 7), np.max(self.stream_data['phi1'] + 7), 1000)
        ax.plot(orbit_phi1, ointerps[track](orbit_phi1), color='red', label='', zorder=0)

    def add_spline_track(self, ax, med_ind=2, label='', color='blue'):
        """
        Add the spline track to the given axis. made to work with vis_6_panel
        """
        importlib.reload(stream_funcs)
        spline_points = stream_funcs.apply_spline(
            np.linspace(np.min(self.stream_data['phi1'] - 5), np.max(self.stream_data['phi1'] + 5), 1000),
            self.spline_points_dict['phi1_spline_points'],
            self.nested_dict['meds'][med_ind],
            self.spline_points_dict['spline_k']
        )
        ax.plot(np.linspace(np.min(self.stream_data['phi1'] - 5), np.max(self.stream_data['phi1'] + 5), 1000), 10**spline_points, color='blue', ls='-.', label='', zorder=0)
        nested_list_exp_meds = self.nested_dict['exp_meds'][med_ind]
        nested_list_exp_ep = self.nested_dict['exp_ep'][med_ind]
        nested_list_exp_em = self.nested_dict['exp_em'][med_ind]
        ax.errorbar(self.spline_points_dict['phi1_spline_points'], np.array(nested_list_exp_meds), yerr=(nested_list_exp_em, nested_list_exp_ep),
            capsize=5, elinewidth=0.75, ecolor=color, ms=6, fmt='o', mfc=color, mec=color, label=label, zorder=0)

    def return_spline_track(self, med_ind=0):
        """
        Return the spline track for the given median index for more flexible plotting. Made for working with lsigv and pstream
        """
        importlib.reload(stream_funcs)
        spline_points = stream_funcs.apply_spline(
            np.linspace(np.min(self.stream_data['phi1'] - 5), np.max(self.stream_data['phi1'] + 5), 1000),
            self.spline_points_dict['phi1_spline_points'],
            self.nested_dict['meds'][med_ind],
            self.spline_points_dict['spline_k']
        )
        lsigv_phi1_spline_points = self.spline_points_dict['lsigv_phi1_spline_points']
        nested_list_exp_meds = self.nested_dict['exp_meds'][med_ind]
        nested_list_exp_ep = self.nested_dict['exp_ep'][med_ind]
        nested_list_exp_em = self.nested_dict['exp_em'][med_ind]
        return {
            'phi1': np.linspace(np.min(self.stream_data['phi1'] - 5), np.max(self.stream_data['phi1'] + 5), 1000),
            'spline': 10**spline_points,
            'lsigv_phi1_spline_points': lsigv_phi1_spline_points,
            'meds': nested_list_exp_meds,
            'ep': nested_list_exp_ep,
            'em': nested_list_exp_em
        }
    

    def vis_6_panel(self, addBackground=True, save_fig=False, fig_path=None, dist_mod_panel=False, **kwargs):
        """
        Visualize the 6 panel plot for the stream data.
        """
        highlight = kwargs.get('highlight', None)
        pad = kwargs.get('pad', None)  # Default padding values
        x_arr = np.linspace((self.spline_points_dict['phi1_spline_points'][0]-1), (self.spline_points_dict['phi1_spline_points'][-1]+5), 100)
        
        # Decide panel layout: include distance-modulus panel or not
        stream_data = self.stream_data
        requested_dm = bool(dist_mod_panel)
        # Support both Astropy Table (colnames) and pandas DataFrame (columns)
        has_dm = (
            (hasattr(stream_data, 'colnames') and ('dist_mod' in stream_data.colnames)) or
            (hasattr(stream_data, 'columns') and ('dist_mod' in getattr(stream_data, 'columns')))
        )
        include_dm_panel = requested_dm and has_dm
        if requested_dm and not has_dm:
            print('dist_mod not found in stream_data; skipping DM panel.')
        n_pan = 6 if include_dm_panel else 5
        fig, ax = plt.subplots(n_pan, 1, figsize=(15, 2.5 * n_pan), sharex=True)
        # axis indices
        phi2_ax_i, vgsr_ax_i, pmra_ax_i, pmdec_ax_i = 0, 1, 2, 3
        dm_ax_i = 4 if include_dm_panel else None
        feh_ax_i = 5 if include_dm_panel else 4
        cmap = 'viridis'
        cm = ax[phi2_ax_i].scatter(stream_data['phi1'], stream_data['phi2'], s=30, edgecolor='k', linewidth=0.75,cmap=cmap, c=stream_data['stream_prob'], alpha=1, zorder=1, vmin=0.5, vmax=1)
        ax[vgsr_ax_i].scatter(stream_data['phi1'], stream_data['VGSR'], s=30, edgecolor='k', linewidth=0.75,cmap=cmap, c=stream_data['stream_prob'], alpha=1, zorder=1, vmin=0.5, vmax=1)
        ax[pmra_ax_i].scatter(stream_data['phi1'], stream_data['PMRA'], s=30, edgecolor='k', linewidth=0.75,cmap=cmap, c=stream_data['stream_prob'], alpha=1, zorder=1, vmin=0.5, vmax=1)
        ax[pmdec_ax_i].scatter(stream_data['phi1'], stream_data['PMDEC'], s=30, edgecolor='k', linewidth=0.75,cmap=cmap, c=stream_data['stream_prob'], alpha=1, zorder=1, vmin=0.5, vmax=1)
        if include_dm_panel and dm_ax_i is not None:
            # Distance modulus panel
            ax[dm_ax_i].scatter(stream_data['phi1'], stream_data['dist_mod'], s=30, edgecolor='k', linewidth=0.75, cmap=cmap, c=stream_data['stream_prob'], alpha=1, zorder=1, vmin=0.5, vmax=1)
        ax[feh_ax_i].scatter(stream_data['phi1'], stream_data['FEH'], s=30, edgecolor='k', linewidth=0.75,cmap=cmap, c=stream_data['stream_prob'], alpha=1, zorder=1, vmin=0.5, vmax=1)
        if highlight is not None:
            ax[phi2_ax_i].scatter(highlight['phi1'], highlight['phi2'], s=80, c='m', alpha=0.5, label='highlight')
            ax[vgsr_ax_i].scatter(highlight['phi1'], highlight['VGSR'], s=80, c='m', alpha=0.5, label='highlight')
            ax[pmra_ax_i].scatter(highlight['phi1'], highlight['PMRA'], s=80, c='m', alpha=0.5, label='highlight')
            ax[pmdec_ax_i].scatter(highlight['phi1'], highlight['PMDEC'], s=80, c='m', alpha=0.5, label='highlight')
            if include_dm_panel and dm_ax_i is not None:
                ax[dm_ax_i].scatter(highlight['phi1'], highlight['dist_mod'], s=80, c='m', alpha=0.5, label='highlight')
            ax[feh_ax_i].scatter(highlight['phi1'], highlight['FEH'], s=80, c='m', alpha=0.5, label='highlight')

        if pad is not None:
            ax[phi2_ax_i].axhline(0, color='k', lw=1, zorder=2)
            ax[phi2_ax_i].axhline(pad[4], color='k', lw=0.5, ls='-.', zorder=0)
            ax[phi2_ax_i].axhline(-pad[4], color='k', lw=0.5, ls='-.', zorder=0)

            ax[vgsr_ax_i].plot(x_arr, apply_spline(x_arr, self.spline_points_dict['phi1_spline_points'], self.nested_dict['meds'][self.vgsr_idx], self.spline_points_dict['spline_k']), color='k', lw=1, zorder=0)
            ax[vgsr_ax_i].plot(x_arr, apply_spline(x_arr, self.spline_points_dict['phi1_spline_points'], self.nested_dict['meds'][self.vgsr_idx], self.spline_points_dict['spline_k'])+pad[0], color='k', lw=0.5, ls='-.', zorder=0)
            ax[vgsr_ax_i].plot(x_arr, apply_spline(x_arr, self.spline_points_dict['phi1_spline_points'], self.nested_dict['meds'][self.vgsr_idx], self.spline_points_dict['spline_k'])-pad[0], color='k', lw=0.5, ls='-.', zorder=0)

            ax[pmra_ax_i].plot(x_arr, apply_spline(x_arr, self.spline_points_dict['phi1_spline_points'], self.nested_dict['meds'][self.pmra_idx], self.spline_points_dict['spline_k']), color='k', lw=1, zorder=0)
            ax[pmra_ax_i].plot(x_arr, apply_spline(x_arr, self.spline_points_dict['phi1_spline_points'], self.nested_dict['meds'][self.pmra_idx], self.spline_points_dict['spline_k'])+pad[2], color='k', lw=0.5, ls='-.', zorder=0)
            ax[pmra_ax_i].plot(x_arr, apply_spline(x_arr, self.spline_points_dict['phi1_spline_points'], self.nested_dict['meds'][self.pmra_idx], self.spline_points_dict['spline_k'])-pad[2], color='k', lw=0.5, ls='-.', zorder=0)

            ax[pmdec_ax_i].plot(x_arr, apply_spline(x_arr, self.spline_points_dict['phi1_spline_points'], self.nested_dict['meds'][self.pmdec_idx], self.spline_points_dict['spline_k']), color='k', lw=1, zorder=0)
            ax[pmdec_ax_i].plot(x_arr, apply_spline(x_arr, self.spline_points_dict['phi1_spline_points'], self.nested_dict['meds'][self.pmdec_idx], self.spline_points_dict['spline_k'])+pad[3], color='k', lw=0.5, ls='-.', zorder=0)
            ax[pmdec_ax_i].plot(x_arr, apply_spline(x_arr, self.spline_points_dict['phi1_spline_points'], self.nested_dict['meds'][self.pmdec_idx], self.spline_points_dict['spline_k'])-pad[3], color='k', lw=0.5, ls='-.', zorder=0)

            ax[feh_ax_i].axhline(self.nested_dict['meds'][3], color='k', lw=1, zorder=0)
            ax[feh_ax_i].axhline(pad[1], color='k', lw=0.5, ls='-.', zorder=0)

        ax[vgsr_ax_i].errorbar(
            stream_data['phi1'], stream_data['VGSR'],
            yerr=stream_data['VRAD_ERR'],
            capsize=0, elinewidth=0.75, ecolor='k', ms=6, fmt='none', mfc='none', mec='none', zorder=0
        )
        ax[pmra_ax_i].errorbar(
            stream_data['phi1'], stream_data['PMRA'],
            yerr=stream_data['PMRA_ERROR'],
            capsize=0, elinewidth=0.75, ecolor='k', ms=6, fmt='none', mfc='none', mec='none', zorder=0
        )
        ax[pmdec_ax_i].errorbar(
            stream_data['phi1'], stream_data['PMDEC'],
            yerr=stream_data['PMDEC_ERROR'],
            capsize=0, elinewidth=0.75, ecolor='k', ms=6, fmt='none', mfc='none', mec='none', zorder=0
        )
        if include_dm_panel and dm_ax_i is not None:
            # DM errorbars: support plus/minus or symmetric
            if 'dist_mod_err_plus' in stream_data.colnames and 'dist_mod_err_minus' in stream_data.colnames:
                yerr_dm = (stream_data['dist_mod_err_minus'], stream_data['dist_mod_err_plus'])
            else:
                yerr_dm = stream_data['dist_mod_err'] if 'dist_mod_err' in stream_data.colnames else None
            try:
                ax[dm_ax_i].errorbar(
                    stream_data['phi1'], stream_data['dist_mod'],
                    yerr=yerr_dm,
                    capsize=0, elinewidth=0.75, ecolor='k', ms=6, fmt='none', mfc='none', mec='none', zorder=0
                )
            except Exception:
                pass
        ax[feh_ax_i].errorbar(
            stream_data['phi1'], stream_data['FEH'],
            yerr=stream_data['FEH_ERR'],
            capsize=0, elinewidth=0.75, ecolor='k', ms=6, fmt='none', mfc='none', mec='none', zorder=0
        )
        ax[phi2_ax_i].set_ylabel(r'$\phi_2$ (deg)', fontsize=14)
        ax[vgsr_ax_i].set_ylabel(r'$V_{GSR}$ (km/s)', fontsize=14)
        ax[pmra_ax_i].set_ylabel(r'$\mu_{\alpha}$ (mas/yr)', fontsize=14)
        ax[pmdec_ax_i].set_ylabel(r'$\mu_{\delta}$ (mas/yr)', fontsize=14)
        if include_dm_panel and dm_ax_i is not None:
            ax[dm_ax_i].set_ylabel('Distance modulus (mag)', fontsize=14)
        ax[feh_ax_i].set_ylabel(r'[Fe/H]', fontsize=14)
        ax[feh_ax_i].set_xlabel(r'$\phi_1$ (deg)', fontsize=14)

        if addBackground:
            ax[phi2_ax_i].scatter(self.all_memberships['phi1'], self.all_memberships['phi2'], s=10, color='k', edgecolors='none', alpha=0.05, zorder=0)
            ax[vgsr_ax_i].scatter(self.all_memberships['phi1'], self.all_memberships['VGSR'], s=10, color='k', edgecolors='none', alpha=0.05, zorder=0)
            ax[pmra_ax_i].scatter(self.all_memberships['phi1'], self.all_memberships['PMRA'], s=10, color='k', edgecolors='none', alpha=0.05, zorder=0)
            ax[pmdec_ax_i].scatter(self.all_memberships['phi1'], self.all_memberships['PMDEC'], s=10, color='k', edgecolors='none', alpha=0.05, zorder=0)
            if include_dm_panel and dm_ax_i is not None:
                ax[dm_ax_i].scatter(self.all_memberships['phi1'], self.all_memberships['dist_mod'], s=10, color='k', edgecolors='none', alpha=0.05, zorder=0)
            ax[feh_ax_i].scatter(self.all_memberships['phi1'], self.all_memberships['FEH'], s=10, color='k', edgecolors='none', alpha=0.05, zorder=0)

        # xlim based on the phi1 values
        ax[phi2_ax_i].set_xlim(np.min(stream_data['phi1']) - 2, np.max(stream_data['phi1']) + 2)

        # set ylims based on stream data y values
        ax[phi2_ax_i].set_ylim(np.min(stream_data['phi2']) - 2, np.max(stream_data['phi2']) + 2)
        ax[vgsr_ax_i].set_ylim(np.min(stream_data['VGSR']) - 10, np.max(stream_data['VGSR']) + 10)
        ax[pmra_ax_i].set_ylim(np.min(stream_data['PMRA']) - 1, np.max(stream_data['PMRA']) + 1)
        ax[pmdec_ax_i].set_ylim(np.min(stream_data['PMDEC']) - 1, np.max(stream_data['PMDEC']) + 1)
        if include_dm_panel and dm_ax_i is not None:
            try:
                dm_vals = np.asarray(stream_data['dist_mod'])
                dm_pad = 0.2
                ax[dm_ax_i].set_ylim(np.nanmin(dm_vals) - dm_pad, np.nanmax(dm_vals) + dm_pad)
            except Exception:
                pass

        if hasattr(self, 'results_o'):
            self.add_orbit_track(ax[phi2_ax_i], self.results_o, track='phi2')
            self.add_orbit_track(ax[vgsr_ax_i], self.results_o, track='vgsr')
            self.add_orbit_track(ax[pmra_ax_i], self.results_o, track='pmra')
            self.add_orbit_track(ax[pmdec_ax_i], self.results_o, track='pmdec')
            # For distance modulus panel we skip orbit distance overlay (units mismatch). If you prefer, we can convert dist->dm.
        else:
            print('No orbit results found, skipping orbit track plotting')
        
        ax[vgsr_ax_i].plot(x_arr, apply_spline(x_arr, self.spline_points_dict['phi1_spline_points'], self.nested_dict['meds'][self.vgsr_idx], self.spline_points_dict['spline_k']), color='b', lw=1, zorder=0, ls='--')
        xsp = np.asarray(self.spline_points_dict['phi1_spline_points'])
        y_v = np.asarray(self.nested_dict['exp_meds'][1])
        ym_v = np.asarray(self.nested_dict['exp_em'][1])
        yp_v = np.asarray(self.nested_dict['exp_ep'][1])
        if y_v.ndim == 0: y_v = y_v[None]
        if ym_v.ndim == 0: ym_v = ym_v[None]
        if yp_v.ndim == 0: yp_v = yp_v[None]
        nv = int(min(len(xsp), len(y_v), len(ym_v), len(yp_v)))
        if nv > 0:
            ax[vgsr_ax_i].errorbar(xsp[:nv], y_v[:nv], yerr=(ym_v[:nv], yp_v[:nv]),
                capsize=5, elinewidth=0.75, ecolor='b', ms=2, fmt='o', mfc='b', mec='b', zorder=0)
        ax[pmra_ax_i].plot(x_arr, apply_spline(x_arr, self.spline_points_dict['phi1_spline_points'], self.nested_dict['meds'][self.pmra_idx], self.spline_points_dict['spline_k']), color='b', lw=1, zorder=0, ls='--')
        y_r = np.asarray(self.nested_dict['exp_meds'][5])
        ym_r = np.asarray(self.nested_dict['exp_em'][5])
        yp_r = np.asarray(self.nested_dict['exp_ep'][5])
        if y_r.ndim == 0: y_r = y_r[None]
        if ym_r.ndim == 0: ym_r = ym_r[None]
        if yp_r.ndim == 0: yp_r = yp_r[None]
        nr = int(min(len(xsp), len(y_r), len(ym_r), len(yp_r)))
        if nr > 0:
            ax[pmra_ax_i].errorbar(xsp[:nr], y_r[:nr], yerr=(ym_r[:nr], yp_r[:nr]),
                capsize=5, elinewidth=0.75, ecolor='b', ms=2, fmt='o', mfc='b', mec='b', zorder=0)
        ax[pmdec_ax_i].plot(x_arr, apply_spline(x_arr, self.spline_points_dict['phi1_spline_points'], self.nested_dict['meds'][self.pmdec_idx], self.spline_points_dict['spline_k']), color='b', lw=1, zorder=0, ls='--')
        y_d = np.asarray(self.nested_dict['exp_meds'][6])
        ym_d = np.asarray(self.nested_dict['exp_em'][6])
        yp_d = np.asarray(self.nested_dict['exp_ep'][6])
        if y_d.ndim == 0: y_d = y_d[None]
        if ym_d.ndim == 0: ym_d = ym_d[None]
        if yp_d.ndim == 0: yp_d = yp_d[None]
        nd = int(min(len(xsp), len(y_d), len(ym_d), len(yp_d)))
        if nd > 0:
            ax[pmdec_ax_i].errorbar(xsp[:nd], y_d[:nd], yerr=(ym_d[:nd], yp_d[:nd]),
                capsize=5, elinewidth=0.75, ecolor='b', ms=2, fmt='o', mfc='b', mec='b', zorder=0)
        ax[feh_ax_i].plot(x_arr, apply_spline(x_arr, self.spline_points_dict['phi1_spline_points'], self.nested_dict['meds'][3], self.spline_points_dict['spline_k']), color='b', lw=1, zorder=0, ls='--')
        cbar = fig.colorbar(cm, ax=ax, orientation='vertical', pad=0.01, aspect=50)
        cbar.set_label('Membership Probability', fontsize=12)
        cbar.ax.tick_params(labelsize=12)
        for a in ax:
            plot_form(a)

#
        for ax in plt.gcf().get_axes():
            for artist in ax.get_children():
                artist.set_rasterized(True)
        if save_fig:
            if fig_path is None:
                fig_path = 'figures_draft/postmcmc_6panel.pdf'
            plt.savefig(fig_path, bbox_inches='tight', dpi=600)
        plt.show()
        
        #self.add_spline_track(ax[5], med_ind=3, color='blue')

    def GaiaDECALS_cut(self, pad=[], GaiaDeCALS_path = '', upper_bound_feh=True, MaskList=['VGSR', 'FEH', 'PMRA', 'PMDEC', 'phi2'], save_fig=False, fig_path=None, addOrbit=False, useBox=False):
        if pad is None:
            if not hasattr(self, 'pad'):
                print('No padding has been provided, either pass your own or run box_cut first')
            else:
                pad = self.pad

        self.GaiaDeCALS = GaiaDeCALSHandler(self.frame, self.spline_points_dict, self.nested_dict, GaiaDeCALS_data=table.Table.read(GaiaDeCALS_path).to_pandas(), stream_data=self.stream_data, isochrone_path=self.isochrone_path, min_dist=self.min_dist)
        self.GaiaDeCALS.apply_box_cut()
        self.GaiaDeCALS.apply_iso_cut()
        self.GaiaDeCALS.apply_phot_metallicity()

        gaia_decals = self.GaiaDeCALS.data[self.GaiaDeCALS.sel_all]
        #sel_qual = get_sel_qual_mask()

        #desi_match_all = gaia_decals[np.isin(gaia_decals['source_id'],self.desi_data['SOURCE_ID'])]
        #desi_match_qual = gaia_decals[np.isin(gaia_decals['source_id'],self.desi_data['SOURCE_ID'][sel_qual])]
        if useBox:
            stream_data = self.box_data
        else:
            stream_data = self.stream_data
        desi_match_prob = gaia_decals[np.isin(gaia_decals['source_id'],stream_data['SOURCE_ID'])]

        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        ax1.scatter(gaia_decals['phi1'], gaia_decals['phi2'], marker='s',facecolor='0.2',  s=35, linewidth=0.75, zorder=0, edgecolor='tab:blue', label='Gaia+DeCaLs', alpha=0.25)
        #ax.scatter(desi_match_all['phi1'], desi_match_all['phi2'],marker='s',facecolor='none', s=55, linewidth=0.75, zorder=0, edgecolor='k', label='In DESI, no quality cut', alpha=0.9)
        #ax1.scatter(desi_match_qual['phi1'], desi_match_qual['phi2'],marker='s',facecolor='none', s=35, linewidth=0.75, zorder=0, edgecolor='orange', alpha=0.9)
        ax1.scatter(desi_match_prob['phi1'], desi_match_prob['phi2'],marker='s',facecolor='none', s=35, linewidth=0.75, zorder=0, edgecolor='blue',alpha=0.9)

        ax1.scatter(stream_data['phi1'], stream_data['phi2'],marker='s',facecolor='none', edgecolor='blue', s=40, linewidth=0.75, ls=(6, (2, 2)), zorder=0,  alpha=1)
        # put legend above plot
        ax1.legend(prop={'size': 10}, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.2), frameon=False)
        #ax.axhline(-5, c='tab:orange')
        ax1.set_xlim(-10, 39)
        ax1.set_ylim(-10,10)
        # ax1.set_xlabel(r'$\phi_1$ [deg]', fontsize=14)
        ax1.set_ylabel(r'$\phi_2$ [deg]', fontsize=14)

        if addOrbit:
            #check if results_o param exists
            if hasattr(self, 'results_o'):
                self.add_orbit_track(ax1, self.results_o, track='phi2')
                self.add_orbit_track(ax2, self.results_o, track='phi2')
            else:
                print('No orbit has been run yet, please run do_orbit first')
        

        legend_handles = [
            plt.Line2D([0], [0], marker='s', linestyle='none', mec='none', mfc='0.8', label='Gaia+DeCaLs', alpha=1, markersize=7),
            # plt.Line2D([0], [0], marker='s', linestyle='none', mec='orange', mfc='none', label='In DESI, pass quality cuts', markersize=7),
            plt.Line2D([0], [0], marker='s', linestyle='none', mec='blue', mfc='none', label=r'Box Cut Members', markersize=7),
        ]
        ax1.legend(handles=legend_handles, ncol=3, prop={'size': 10}, loc='upper center', bbox_to_anchor=(0.5, 1.22), frameon=False)

        plot_form(ax1)


        ax2.scatter(gaia_decals['phi1'], gaia_decals['phi2'], marker='s',facecolor='0.2', s=35, linewidth=0.75, zorder=0, edgecolor='tab:blue', label='Gaia+DeCaLs', alpha=0.25)
        #ax.scatter(desi_match_all['phi1'], desi_match_all['phi2'],marker='s',facecolor='none', s=55, linewidth=0.75, zorder=0, edgecolor='k', label='In DESI, no quality cut', alpha=0.9)
        #ax.scatter(desi_match_qual['phi1'], desi_match_qual['phi2'],marker='s',facecolor='none', s=55, linewidth=0.75, zorder=0, edgecolor='m', label='In DESI, quality cut', alpha=0.9)
        #ax.scatter(desi_match_prob['phi1'], desi_match_prob['phi2'],marker='s',facecolor='none', s=55, linewidth=0.75.5, zorder=0, edgecolor='tab:orange',label=r'p > 0.5 $\in$ DESI',alpha=0.9)
        #ax.scatter(stream_data['phi1'], stream_data['phi2'],marker='s',facecolor='none', s=60, linewidth=0.75, ls=(6, (2, 2)), zorder=0, edgecolor='tab:orange',  alpha=1)
        # put legend above plot
        #ax2.legend(prop={'size': 8}, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.07), frameon=False)
        ax2.set_xlim(-10, 39)
        ax2.set_ylim(-10,10);
        ax2.set_xlabel(r'$\phi_1$ [deg]', fontsize=14)
        ax2.set_ylabel(r'$\phi_2$ [deg]', fontsize=14)
        plot_form(ax2)
        plt.tight_layout()
        for ax in plt.gcf().get_axes():
            for artist in ax.get_children():
                artist.set_rasterized(True)
        if save_fig:
            if fig_path is None:
                fig_path = 'figures_draft/gaia_decals_desi_stream.pdf'
            plt.savefig(fig_path, bbox_inches='tight', dpi=600)
        plt.show()

    def vis_isochrone(self, color_index_wiggle = 0.18, isochrone_path='/home/jupyter-nassermoha/raid_nassermoha/data/dotter/iso_a13.5_z0.00010.dat', return_axes = False):
        dotter_mp = np.loadtxt(isochrone_path)
        # Obtain the M_g and M_r color band data
        dotter_g_mp = dotter_mp[:,6]
        dotter_r_mp = dotter_mp[:,7]

        bhb_color_wiggle = 0.4
        bhb_abs_mag_wiggle = 0.1

        # build the BHB using empirical data from M92
        dm_m92_harris = 14.59 #dm of M92
        m92ebv = 0.023
        m92ag = m92ebv * 3.184
        m92ar = m92ebv * 2.130
        m92_hb_r = np.array([17.3, 15.8, 15.38, 15.1, 15.05])
        m92_hb_col = np.array([-0.39, -0.3, -0.2, -0.0, 0.1])
        m92_hb_g = m92_hb_r + m92_hb_col
        des_m92_hb_g = m92_hb_g - 0.104 * (m92_hb_g - m92_hb_r) + 0.01
        des_m92_hb_r = m92_hb_r - 0.102 * (m92_hb_g - m92_hb_r) + 0.02
        des_m92_hb_g = des_m92_hb_g - m92ag - dm_m92_harris
        des_m92_hb_r = des_m92_hb_r - m92ar - dm_m92_harris

        if hasattr(self, 'desi_data'):
            self.desi_handler = DataHandler(self.frame, self.spline_points_dict, self.nested_dict)
            self.desi_handler.data = self.desi_data

            self.desi_handler.apply_box_cut(pad=np.array([400. , 0. , 100,  100,  10 ])) # now have self.ms_handler.touch_masks dictionary
            self.desi_handler.apply_plx_cut(min_dist=self.min_dist) # now have self.ms_handler.plx_mask
            self.desi_handler.apply_phi1_mask() # now have self.ms_handler.phi1_mask
            self.desi_handler.all_box_cuts_DESI(withIso=False)


            desi_data = self.desi_handler.data[self.desi_handler.sel]
            desi_ebv = np.array(desi_data['EBV'].values)
            desi_g_flux, desi_r_flux = np.array(desi_data['FLUX_G'].values), np.array(desi_data['FLUX_R'].values)

            # Custom function to calculate the color index (g-r), absolute R magnitude, and apparent r magnitude after EBV correction
            desi_colour_index, desi_abs_mag, desi_r_mag = stream_funcs.get_colour_index_and_abs_mag(desi_ebv, desi_g_flux, desi_r_flux, self.min_dist)
     

            big_iso = isochrone_cut(
                color_indx_wiggle=color_index_wiggle,
                isochrone_path=isochrone_path,
                desi_data=desi_data,
                desi_distance=self.min_dist)
        else:
            print('No desi_data provided, wont use in vis')

        stream_data = self.stream_data
        stream_ebv = np.array(stream_data['EBV'])
        stream_g_flux, stream_r_flux = np.array(stream_data['FLUX_G']), np.array(stream_data['FLUX_R'])
        # Custom function to calculate the color index (g-r), absolute R magnitude, and apparent r magnitude after EBV correction
        stream_colour_index, stream_abs_mag, stream_r_mag = stream_funcs.get_colour_index_and_abs_mag(stream_ebv, stream_g_flux, stream_r_flux, self.min_dist)


        norm = Normalize(vmin=0.5, vmax=1)
        fig, ax = plt.subplots(figsize=(7, 8))

        cm = ax.scatter(stream_colour_index, stream_abs_mag, c=stream_data['stream_prob'], cmap='viridis', norm=norm, edgecolors='k', linewidth=0.8, s=30, label='Stream Members', alpha=1, zorder=1)
        ax.set_xlabel(r'$(g-r)$ [mag]', fontsize=14)
        ax.set_ylabel(r'$r$ [mag]', fontsize=14)

        if hasattr(self, 'desi_data'):
            ax.scatter(desi_colour_index[big_iso], desi_abs_mag[big_iso], c='lightblue', alpha=0.1, s=5, zorder=0)
            ax.scatter(desi_colour_index[~big_iso], desi_abs_mag[~big_iso], c='k', alpha=0.01, s=1, zorder=0)


        ax.plot(dotter_g_mp - dotter_r_mp, dotter_r_mp, c='b', alpha=0.3)
        ax.plot(dotter_g_mp - dotter_r_mp - color_index_wiggle, dotter_r_mp, c='b', alpha=0.3)
        ax.plot(dotter_g_mp - dotter_r_mp + color_index_wiggle, dotter_r_mp, c='b', alpha=0.3)

        # M92 horizontal branch band
        ax.plot(des_m92_hb_g - des_m92_hb_r, des_m92_hb_r, c='b', alpha=0.5)
        ax.plot(des_m92_hb_g - des_m92_hb_r, des_m92_hb_r - bhb_color_wiggle, 'r--', alpha=0.5)
        ax.plot(des_m92_hb_g - des_m92_hb_r, des_m92_hb_r + bhb_color_wiggle, 'r--', alpha=0.5)
        ax.plot(des_m92_hb_g - des_m92_hb_r + bhb_abs_mag_wiggle, des_m92_hb_r, 'r-.', alpha=0.5)
        ax.plot(des_m92_hb_g - des_m92_hb_r - bhb_abs_mag_wiggle, des_m92_hb_r, 'r-.', alpha=0.5)
        cbar = plt.colorbar(cm, ax=ax)
        cbar.set_label('Membership Probability', fontsize=15)
        ax.legend(loc='lower left', fontsize=10)
        ax.set_xlim(-0.3, 1)
        ax.invert_yaxis()
        ax.set_ylim(8, -3)  
        

        plot_form(ax)

    def segment_vdisp(self, segments, green_min=np.inf, green_max=np.inf, prob_cutoff=0.5, useBox=False, withPlot=True, save_fig=False, fig_path=None, externalSpline=False):
        if not useBox:
            stream_data = self.stream_data
            high_prob = stream_data['stream_prob'] > prob_cutoff
            label='GMM'
        else:
            stream_data = self.box_data
            label='Box Cut'
            #high_prob is just a mask that is true for everything
            high_prob = np.ones(len(stream_data), dtype=bool)

        phi1 = stream_data['phi1'][high_prob]
        phi2 = stream_data['phi2'][high_prob]
        vgsr = stream_data['VGSR'][high_prob]
        vrad_err = stream_data['VRAD_ERR'][high_prob]

        rv_samples = {}
        rverr_samples = {}
        mcmc_results = {}
        sigmas = {}
        phi1_spline_points = self.spline_points_dict['phi1_spline_points']
        spline_k = self.spline_points_dict['spline_k']
        nested_list_meds = self.nested_dict['meds']
        # MCMC for each segment
        for i, (pmin, pmax) in enumerate(segments):
            print(f'running segment {pmin} to {pmax} deg')
            mask = (phi1 > pmin) & (phi1 < pmax)
            phi1_seg = phi1[mask]
            rv = vgsr[mask] - apply_spline(phi1_seg, phi1_spline_points, nested_list_meds[self.vgsr_idx], k=spline_k)
            err = vrad_err[mask]

            rv_samples[i] = rv
            rverr_samples[i] = err
            mcmc_results[i] = vdisp.mcmc(rv, err, nsteps=500)
            sigmas[i] = 10**mcmc_results[i][:,1]

        # Print summary stats
        for i, (pmin, pmax) in enumerate(segments):
            mu = mcmc_results[i][:,0]
            sigma = sigmas[i]
            print(f"Segment {pmin} to {pmax} deg:")
            print(f"  mean RV: {np.median(mu):.2f} (+{np.percentile(mu, 84)-np.median(mu):.2f}/-{np.median(mu)-np.percentile(mu, 16):.2f})")
            print(f"  σ_v:     {np.median(sigma):.2f} (+{np.percentile(sigma, 84)-np.median(sigma):.2f}/-{np.median(sigma)-np.percentile(sigma, 16):.2f})")

        # MCMC for full
        rv = vgsr - apply_spline(phi1, phi1_spline_points, nested_list_meds[self.vgsr_idx], k=spline_k)
        err = vrad_err

        rv_sample = rv
        rverr_sample= err
        mcmc_result = vdisp.mcmc(rv, err, nsteps=500)
        sigma = 10**mcmc_result[:,1]

        mu = mcmc_result[:,0]

        print(" ")
        print(f"OVERALL")
        print(f"  mean RV: {np.median(mu):.2f} (+{np.percentile(mu, 84)-np.median(mu):.2f}/-{np.median(mu)-np.percentile(mu, 16):.2f})")
        print(f"  σ_v:     {np.median(sigma):.2f} (+{np.percentile(sigma, 84)-np.median(sigma):.2f}/-{np.median(sigma)-np.percentile(sigma, 16):.2f})")
    

        if withPlot:
            if not hasattr(self, 'results_o'):
                print('No orbit has been run yet, plots will not include orbit track')
            from matplotlib.colors import Normalize

            # Create a new figure with three vertically-stacked subplots sharing the same x-axis
            fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
            fig.subplots_adjust(hspace=0.1, wspace=0.3)  # Reduced vertical spacing
            ax1 = ax[0]  # Top panel
            ax2 = ax[1]  # Middle panel
            ax3 = ax[2]  # Bottom panel


            # Color map and normalization
            color_map = 'seismic'
            min_prob = -40
            max_prob = 40
            norm = Normalize(vmin=min_prob, vmax=max_prob)

            # Top left panel: phi2 vs phi1 color-coded by residual VGSR
            cm = ax1.scatter(phi1, phi2, 
                            c=vgsr - apply_spline(phi1, phi1_spline_points, nested_list_meds[self.vgsr_idx], k=spline_k),
                            cmap=color_map, s=30, linewidth=0.5, edgecolors='k',
                            zorder=1, norm=norm, label='C-19')
            ax1.axhline(-5, color='k', lw=1, ls='dotted', zorder=0, label=r'$\phi_2$ cut', alpha=0.5)
            ax1.axhline(5, color='k', lw=1, ls='dotted', zorder=0, alpha=0.5)
            #ax1.scatter(desi_data['phi1'], desi_data['phi2'], c='0.9', s=10, linewidth=0.5,
            #            edgecolors='k', zorder=0, alpha=0.01, label='Background')


            #ax1.plot(orbit_phi1, ointerps['phi2'](orbit_phi1), color='r', ls='--', label='Orbit', zorder=0)

            ax1.set_ylabel('$\phi_2$ [deg]', fontsize=12)
            ax1.set_xlim(-13, 43)
            ax1.set_ylim(-7, 7)
            ax1.tick_params(direction='in')
            ax1.tick_params(labelbottom=False)  # Hide x-axis tick labels

            # Middle left panel: VGSR vs phi1
            ax2.scatter(phi1, vgsr, 
                        c=vgsr - apply_spline(phi1, phi1_spline_points, nested_list_meds[self.vgsr_idx], k=spline_k),
                        cmap=color_map, s=30, linewidth=0.5, edgecolors='k', zorder=1, norm=norm)
            x_arr = np.linspace(-10, 40, 100)
            ax2.plot(x_arr, apply_spline(x_arr, phi1_spline_points, nested_list_meds[self.vgsr_idx], k=spline_k),
                    c='b', ls='--', lw=1, zorder=0)
            #ax2.plot(orbit_phi1, ointerps['vgsr'](orbit_phi1), color='r', ls='--', label='Orbit', zorder=0)

            ax2.set_ylim(-60, 35)
            ax2.set_ylabel(r'$v_{GSR}$ [km/s]', fontsize=12)
            ax2.tick_params(direction='in')
            ax2.tick_params(labelbottom=False)  # Hide x-axis tick labels
            # Error bars for ax2
            for i, (pmin, pmax) in enumerate(segments):
                mask = (phi1 > pmin) & (phi1 < pmax)
                phi1_seg = phi1[mask]
                vgsr_seg = vgsr[mask]
                err = vrad_err[mask]
                
                # Plot error bars
                ax2.errorbar(phi1_seg, vgsr_seg, yerr=err, fmt='None', color='0', alpha=0.9, zorder=0)

            # Bottom left panel: velocity dispersion vs phi1
            for i, (pmin, pmax) in enumerate(segments):
                x_fill = np.linspace(pmin + 0.2, pmax - 0.2, 100)
                sigma = sigmas[i]
                ax3.plot(x_fill, np.ones_like(x_fill) * np.median(sigma), c='k', lw=1, zorder=1)
                ax3.fill_between(x_fill, np.percentile(sigma, 16), np.percentile(sigma, 84), 
                                color='steelblue', alpha=1, zorder=0)

            ax3.set_ylabel(r'$\sigma_{v}$ [km/s]', fontsize=12)
            ax3.set_ylim(0, 25)
            ax3.tick_params(direction='in')


            #----------------------Decide your phi1 green range----------------------------#
            green_min = green_min
            green_max = green_max
            if green_min == np.inf or green_max == np.inf:
                green_min = np.min(phi1) + 0.2
                green_max = np.max(phi1) - 0.2
            #------------------------------------------------------------------------------#

            mask = (phi1 > green_min) & (phi1 < green_max)
            ax4 = fig.add_axes([1.02, 0.11, 0.21, 0.77])  # corresponding to x position, y position, width, height
            ax4.scatter(phi2[mask], vgsr[mask] - apply_spline(phi1[mask], phi1_spline_points, nested_list_meds[self.vgsr_idx], k=spline_k), cmap=color_map, 
                        c=vgsr[mask] - apply_spline(phi1[mask], phi1_spline_points, nested_list_meds[self.vgsr_idx], k=spline_k),
                        s=30, linewidth=0.5, edgecolors='k', zorder=1, norm=norm)
            ax4.errorbar(phi2[mask], vgsr[mask] - apply_spline(phi1[mask], phi1_spline_points, nested_list_meds[self.vgsr_idx], k=spline_k), yerr=vrad_err[mask], fmt='None', color='0', 
                        alpha=0.9, zorder=0)

            ax4.set_xlim(-7, 7) # hard coded
            ax4.set_ylim(-30, 25) # hard coded
            ax4.set_ylabel(r'$\Delta v_{GSR}$ [km/s]', fontsize=12)
            ax4.tick_params(axis='x', colors='white')
            ax4.set_xlabel(r'$\phi_2$ [deg]', fontsize=12)
            ax4.tick_params(direction='in')




            # Colorbar
            cbar_ax = fig.add_axes([1.27, 0.101, 0.025, 0.78])  # Adjust the position of the colorbar
            cbar = fig.colorbar(cm, cax=cbar_ax)
            cbar.set_label(r'$\Delta v_{GSR}$ [km/s]', fontsize=12)

            # Grid and styling for all subplots
            for a in [ax1, ax2, ax3, ax4]:
                a.grid(ls='-.', alpha=0.2, zorder=0)
                a.tick_params(direction='in')
                a.spines['top'].set_linewidth(1)
                a.spines['right'].set_linewidth(1)
                a.spines['left'].set_linewidth(1)
                a.spines['bottom'].set_linewidth(1)
                a.tick_params(axis='both', which='both', direction='in', top=True, right=True)
                a.minorticks_on()


            for a in [ax2]:
                # Draw at the bottom (y=0 in axis coordinates)
                a.plot([green_min, green_max], [0, 0], transform=a.get_xaxis_transform(), color='green', linewidth=6, alpha=0.8)
                a.plot([green_min, green_max], [1, 1], transform=a.get_xaxis_transform(), color='green', linewidth=6, alpha=0.8)


            ax4.plot([0, 1], [0, 0], transform=ax4.transAxes, color='green', linewidth=6, alpha=0.8)  # bottom edge
            ax4.plot([0, 1], [1, 1], transform=ax4.transAxes, color='green', linewidth=6, alpha=0.8)  # top edge
            ax4.plot([0, 0], [0, 1], transform=ax4.transAxes, color='green', linewidth=6, alpha=0.8)  # left edge
            ax4.plot([1, 1], [0, 1], transform=ax4.transAxes, color='green', linewidth=6, alpha=0.8)  # right edge

            ax3.set_xlabel(r'$\phi_1$ [deg]', fontsize=12)
            # Create legend
            legend_handles = {
                'C-19': plt.Line2D([0], [0], marker='o', color='w', label='C-19 ' + label, 
                                    markerfacecolor='b', markersize=8, linestyle='None'),
                'phi2': plt.Line2D([0], [0], marker='none', color='k', label=r'$\phi_2$ cut',
                                        linestyle='dotted'),
                'Orbit': plt.Line2D([0], [0], color='r', lw=1, ls='solid',label='Orbit')
            }
            ax1.legend(handles=legend_handles.values(), loc='upper right', fontsize=8, frameon=False)
            legend_handles = {
                'Spline' : plt.Line2D([0], [0], color='b',ls='solid', lw=1, label='Spline')}
            ax2.legend(handles=legend_handles.values(), loc='upper right', fontsize=8, frameon=False)

            if hasattr(self, 'results_o'):
                self.add_orbit_track(ax1, self.results_o, track='phi2')
                self.add_orbit_track(ax2, self.results_o, track='vgsr')

            # Title for the top-left plot
            #ax1.set_title(f'C-19, p> {p}' , fontsize=12)

            if externalSpline:
                return fig, ax

            else: 
                if save_fig:
                    for ax in plt.gcf().get_axes():
                        for artist in ax.get_children():
                            artist.set_rasterized(True)
                    if fig_path is None:
                        fig_path = 'figures_draft/vdispbox.pdf'
                    plt.savefig(fig_path, bbox_inches='tight', dpi=600)


                plt.show()




    def print_meds(self):
        """
        Print the median and error values for the MCMC parameters.

        From Joseph's stream_funtions.py
        """
        stream_dir = self.stream_run_directory
        mcmc_dict = np.load(stream_dir + '/mcmc_dict.npy', allow_pickle=True).item()

        flatchain = mcmc_dict['flatchain']
        meds, errs = process_chain(flatchain, labels = mcmc_dict['extended_param_labels'])
        exp_flatchain = np.copy(flatchain)
        for i, label in enumerate(meds.keys()):
            if label[0] == 'l':
                exp_flatchain[:,i]= 10 ** exp_flatchain[:,i]
        exp_meds, exp_errs = process_chain(exp_flatchain, mcmc_dict['extended_param_labels'])

        _, ep, em = process_chain(mcmc_dict['flatchain'], avg_error=False, labels = mcmc_dict['extended_param_labels'])

        exp_flatchain = np.copy(flatchain)
        for i, label in enumerate(meds.keys()):
            if label[0] == 'l':
                exp_flatchain[:,i]= 10 ** exp_flatchain[:,i]
        exp_meds, exp_ep, exp_em = process_chain(exp_flatchain, avg_error=False, labels = mcmc_dict['extended_param_labels'])

        i = 0
        # print("{:<10} {:>10} {:>10} {:>10} {:>10}".format('param','med','err','exp(med)','exp(err)'))
        print("{:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format('param','med', 'em','ep','exp(med)', 'exp(em)','exp(ep)'))
        print('--------------------------------------------------------------------------------------')
        for label,v in meds.items():
            # if label[:8] == 'lpstream':
            #     print("{:<10} {:>10.3f} {:>10.3f} {:>10.5f} {:>10.5f}".format(label,v,errs[label], np.e**v, np.log(10)*(np.e**v)*errs[label]))
            if label[0] == 'l':
                # print("{:<10} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} ".format(label,v,errs[label], exp_meds[label], exp_errs[label]))
                print("{:<10} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f}".format(label,v,em[label],ep[label], exp_meds[label], exp_em[label], exp_ep[label]))
            else:
                print("{:<10} {:>10.3f} {:>10.3f} {:>10.3f}".format(label, v, em[label], ep[label]))
            i += 1

class streamCompare:
    """
    Class to handle comparison between two streams
    """
    def __init__(self, stream1, stream2, fr=None):
        self.fr = fr

        # rewrite their phi1s to be on the same frame
        if fr is not None:
            print('Putting streams onto same stream frame')
            stream1.stream_data['phi1'], stream1.stream_data['phi2'] = stream_funcs.ra_dec_to_phi1_phi2(fr, np.array(stream1.stream_data['TARGET_RA'])*u.deg, np.array(stream1.stream_data['TARGET_DEC'])*u.deg)
            stream2.stream_data['phi1'], stream2.stream_data['phi2'] = stream_funcs.ra_dec_to_phi1_phi2(fr, np.array(stream2.stream_data['TARGET_RA'])*u.deg, np.array(stream2.stream_data['TARGET_DEC'])*u.deg)
        self.stream1 = stream1
        self.stream2 = stream2
        self.stream1_data = stream1.stream_data
        self.stream2_data = stream2.stream_data
        self.vgsr_idx = stream1.vgsr_idx
        self.pmra_idx = stream1.pmra_idx
        self.pmdec_idx = stream1.pmdec_idx

    def on_sky(self, return_axes=False, **kwargs):
        stream1_label = kwargs.get('stream1_name', 'Stream 1')
        stream2_label = kwargs.get('stream2_name', 'Stream 2')
        title = kwargs.get('title', '')
        
        if self.fr is not None:
            print('original stream frames being used...')
        stream1_data = self.stream1_data
        stream2_data = self.stream2_data
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.scatter(stream1_data['phi1'], stream1_data['phi2'], marker=10, s=40, label=stream1_label + f' ({len(stream1_data)})', color='tab:blue')
        ax.scatter(stream2_data['phi1'], stream2_data['phi2'], marker=11, s=40, label=stream2_label + f' ({len(stream2_data)})', color='tab:orange')

        # phi1 min and max based off the minimum and maximum from combined streams
        phi1_min = min(np.min(stream1_data['phi1']), np.min(stream2_data['phi1']))
        phi1_max = max(np.max(stream1_data['phi1']), np.max(stream2_data['phi1']))
        phi2_min = min(np.min(stream1_data['phi2']), np.min(stream2_data['phi2']))
        phi2_max = max(np.max(stream1_data['phi2']), np.max(stream2_data['phi2']))
        ax.set_xlim(phi1_min - 1, phi1_max + 1)
        ax.set_ylim(phi2_min - 1, phi2_max + 1)
        ax.set_title(title)

        # axes labels
        ax.set_xlabel(r'$\phi_1$ [deg]', fontsize=14)
        ax.set_ylabel(r'$\phi_2$ [deg]', fontsize=14)
        ax.legend()

        plot_form(ax)

        if return_axes:
            return fig, ax
        else:
            plt.show()

    def sixD(self, return_axes=False, **kwargs):
        stream1_label = kwargs.get('stream1_name', 'Stream 1')
        stream2_label = kwargs.get('stream2_name', 'Stream 2')
        title = kwargs.get('title', '')
        if self.fr is not None:
            print('original stream frames being used...')
        stream1_data = self.stream1_data
        stream2_data = self.stream2_data
        fig, ax = plt.subplots(1, 6, figsize=(15, 3), sharey=True, sharex=True)
        ax[0].scatter(stream1_data['phi1'], stream1_data['phi2'], marker=10, s=40, label=stream1_label + f' ({len(stream1_data)})', color='tab:blue')
        ax[0].scatter(stream2_data['phi1'], stream2_data['phi2'], marker=11, s=40, label=stream2_label + f' ({len(stream2_data)})', color='tab:orange')
        ax[0].set_title(title)
        #plot respective spline   ax[1].plot(x_arr, apply_spline(x_arr, self.spline_points_dict['phi1_spline_points'], self.nested_dict['meds'][self.vgsr_idx], self.spline_points_dict['spline_k']), color='b', lw=1, zorder=0, ls='--')
        ax[0].set_ylabel(r'$\phi_2$ [deg]', fontsize=14)
        ax[0].legend()

        
        ax[1].scatter(stream1_data['phi1'], stream1_data['VRAD'], marker=10, s=40, label=stream1_label + f' ({len(stream1_data)})', color='tab:blue')
        ax[1].scatter(stream2_data['phi1'], stream2_data['VRAD'], marker=11, s=40, label=stream2_label + f' ({len(stream2_data)})', color='tab:orange')
        ax[1].set_ylabel(r'$V_{rad}$ [km/s]', fontsize=14)

        for a in ax:
            plot_form(a)

        if return_axes:
            return fig, ax
        else:
            plt.show()

    def six6(self, addBackground=False, save_fig=False, fig_path=None, return_axes=False, residual=0, dist_mod_panel=False, cmap='viridis', **kwargs):
        """
        Visualize the 6 panel plot comparing two streams similar to vis_6_panel but for stream comparison.

        New argument:
        - residual : int {0,1,2}
            0 -> plot raw quantities (default)
            1 -> plot residuals relative to stream1 spline (VGSR, PMRA, PMDEC)
            2 -> plot residuals relative to stream2 spline (VGSR, PMRA, PMDEC)
        - dist_mod_panel : bool, default False
            If True, include a panel showing distance modulus (mag) vs phi1.
            If False (default), omit the distance/distance-modulus panel entirely (reduces panel count by one).
            This is useful when distance modulus is not available for one or both streams.
        - cmap : str or Colormap, default 'viridis'
            Colormap to use when coloring points by membership probability.
        """
        stream1_label = kwargs.get('stream1_name', 'Stream 1')
        stream2_label = kwargs.get('stream2_name', 'Stream 2')
        title = kwargs.get('title', '')
        
        stream1_data = self.stream1_data
        stream2_data = self.stream2_data
        
        # Decide subplot layout depending on whether the distance-modulus panel is requested
        include_dm_panel = bool(dist_mod_panel)
        n_pan = 6 if include_dm_panel else 5
        fig, ax = plt.subplots(n_pan, 1, figsize=(15, 2.5 * n_pan), sharex=True)
        # Index helpers
        phi2_ax_i, vgsr_ax_i, pmra_ax_i, pmdec_ax_i = 0, 1, 2, 3
        dm_ax_i = 4 if include_dm_panel else None
        feh_ax_i = 5 if include_dm_panel else 4
        
        # Prepare distance-modulus data only if we are going to plot that panel
        if include_dm_panel:
            # Guard against missing columns; if missing, gracefully hide the panel by reducing layout
            if ('dist_mod' not in stream1_data.colnames) or ('dist_mod' not in stream2_data.colnames):
                # Rebuild figure without the DM panel to avoid errors
                plt.close(fig)
                include_dm_panel = False
                n_pan = 5
                fig, ax = plt.subplots(n_pan, 1, figsize=(15, 2.5 * n_pan), sharex=True)
                phi2_ax_i, vgsr_ax_i, pmra_ax_i, pmdec_ax_i = 0, 1, 2, 3
                dm_ax_i = None
                feh_ax_i = 4
            else:
                # Distance modulus values and (symmetric) uncertainties (mag)
                dm1 = stream1_data['dist_mod']
                dm2 = stream2_data['dist_mod']
                # Prefer explicit plus/minus if present, else fall back to single error column
                if 'dist_mod_err_plus' in stream1_data.colnames and 'dist_mod_err_minus' in stream1_data.colnames:
                    dm1_err = (stream1_data['dist_mod_err_minus'], stream1_data['dist_mod_err_plus'])
                else:
                    dm1_err = stream1_data['dist_mod_err'] if 'dist_mod_err' in stream1_data.colnames else None
                if 'dist_mod_err_plus' in stream2_data.colnames and 'dist_mod_err_minus' in stream2_data.colnames:
                    dm2_err = (stream2_data['dist_mod_err_minus'], stream2_data['dist_mod_err_plus'])
                else:
                    dm2_err = stream2_data['dist_mod_err'] if 'dist_mod_err' in stream2_data.colnames else None

        # Decide if residuals should be computed and from which stream
        use_residual = residual in (1, 2)
        spline_ref = None
        nd_ref = None

        # initialize residual arrays to avoid UnboundLocalError when some branches don't assign them
        s1_vgsr = s1_pmra = s1_pmdec = s2_vgsr = s2_pmra = s2_pmdec = None

        if residual == 1:
            if hasattr(self.stream1, 'spline_points_dict') and hasattr(self.stream1, 'nested_dict'):
                spline_ref = self.stream1.spline_points_dict
                nd_ref = self.stream1.nested_dict
            else:
                print("Requested residual=1 but stream1 has no spline; falling back to residual=0")
                use_residual = False
        elif residual == 2:
            if hasattr(self.stream2, 'spline_points_dict') and hasattr(self.stream2, 'nested_dict'):
                spline_ref = self.stream2.spline_points_dict
                nd_ref = self.stream2.nested_dict
            else:
                print("Requested residual=2 but stream2 has no spline; falling back to residual=0")
                use_residual = False

        # helper to safely fetch exp_meds and errors
        def fetch_exp(nd, idx):
            # prefer exp_meds/exp_em/exp_ep if present, otherwise fallback to meds and safe defaults
            if nd is None:
                return None, None, None
            meds_source = nd.get('exp_meds', nd.get('meds'))
            if meds_source is None:
                return None, None, None
            med = meds_source[idx]
            em_list = nd.get('exp_em', [None] * len(nd.get('meds', meds_source)))
            ep_list = nd.get('exp_ep', [None] * len(nd.get('meds', meds_source)))
            em = em_list[idx] if em_list is not None else None
            ep = ep_list[idx] if ep_list is not None else None
            return med, em, ep

        # If using residuals and we have a valid reference spline, compute residual arrays for VGSR, PMRA, PMDEC for both streams
        if use_residual and (spline_ref is not None) and (nd_ref is not None):
            spline_points_ref = spline_ref['phi1_spline_points']
            kref = spline_ref['spline_k']

            def ref_val(phi_arr, meds_ind):
                return apply_spline(phi_arr, spline_points_ref, nd_ref['meds'][meds_ind], kref)

            # residuals for stream1
            s1_vgsr = stream1_data['VGSR'] - ref_val(stream1_data['phi1'], self.vgsr_idx)
            s1_pmra = stream1_data['PMRA'] - ref_val(stream1_data['phi1'], self.pmra_idx)
            s1_pmdec = stream1_data['PMDEC'] - ref_val(stream1_data['phi1'], self.pmdec_idx)
            # residuals for stream2
            s2_vgsr = stream2_data['VGSR'] - ref_val(stream2_data['phi1'], self.vgsr_idx)
            s2_pmra = stream2_data['PMRA'] - ref_val(stream2_data['phi1'], self.pmra_idx)
            s2_pmdec = stream2_data['PMDEC'] - ref_val(stream2_data['phi1'], self.pmdec_idx)
        else:
            # disable residual plotting if we couldn't prepare reference values
            use_residual = False

        # Panel 0: phi2 vs phi1 (unchanged)
        norm = Normalize(vmin=0.5, vmax=1)
        cm = ax[phi2_ax_i].scatter(
            stream1_data['phi1'], stream1_data['phi2'], marker=10, s=40,
            c=stream1_data['stream_prob'], norm=norm, cmap=cmap, alpha=0.8, zorder=1,
            label=stream1_label + f' ({len(stream1_data)})'
        )
        ax[phi2_ax_i].scatter(
            stream2_data['phi1'], stream2_data['phi2'], marker=11, s=40,
            c=stream2_data['stream_prob'], norm=norm, cmap=cmap, alpha=0.8, zorder=1,
            label=stream2_label + f' ({len(stream2_data)})'
        )

        # Panel 1: VGSR vs phi1 (or residual)
        if use_residual:
            ax[vgsr_ax_i].scatter(
                stream1_data['phi1'], s1_vgsr, marker=10, s=40,
                c=stream1_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            ax[vgsr_ax_i].scatter(
                stream2_data['phi1'], s2_vgsr, marker=11, s=40,
                c=stream2_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            # errorbars remain the same vertical scale (use original VRAD_ERR)
            ax[vgsr_ax_i].errorbar(
                stream1_data['phi1'], s1_vgsr,
                yerr=stream1_data['VRAD_ERR'],
                capsize=0, elinewidth=0.75, ecolor='tab:blue', ms=6, fmt='none', zorder=0
            )
            ax[vgsr_ax_i].errorbar(
                stream2_data['phi1'], s2_vgsr,
                yerr=stream2_data['VRAD_ERR'],
                capsize=0, elinewidth=0.75, ecolor='tab:orange', ms=6, fmt='none', zorder=0
            )
        else:
            ax[vgsr_ax_i].scatter(
                stream1_data['phi1'], stream1_data['VGSR'], marker=10, s=40,
                c=stream1_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            ax[vgsr_ax_i].scatter(
                stream2_data['phi1'], stream2_data['VGSR'], marker=11, s=40,
                c=stream2_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            ax[vgsr_ax_i].errorbar(
                stream1_data['phi1'], stream1_data['VGSR'],
                yerr=stream1_data['VRAD_ERR'],
                capsize=0, elinewidth=0.75, ecolor='tab:blue', ms=6, fmt='none', zorder=0
            )
            ax[vgsr_ax_i].errorbar(
                stream2_data['phi1'], stream2_data['VGSR'],
                yerr=stream2_data['VRAD_ERR'],
                capsize=0, elinewidth=0.75, ecolor='tab:orange', ms=6, fmt='none', zorder=0
            )

        # Panel 2: PMRA vs phi1 (or residual)
        if use_residual:
            ax[pmra_ax_i].scatter(
                stream1_data['phi1'], s1_pmra, marker=10, s=40,
                c=stream1_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            ax[pmra_ax_i].scatter(
                stream2_data['phi1'], s2_pmra, marker=11, s=40,
                c=stream2_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            ax[pmra_ax_i].errorbar(
                stream1_data['phi1'], s1_pmra,
                yerr=stream1_data['PMRA_ERROR'],
                capsize=0, elinewidth=0.75, ecolor='tab:blue', ms=6, fmt='none', zorder=0
            )
            ax[pmra_ax_i].errorbar(
                stream2_data['phi1'], s2_pmra,
                yerr=stream2_data['PMRA_ERROR'],
                capsize=0, elinewidth=0.75, ecolor='tab:orange', ms=6, fmt='none', zorder=0
            )
        else:
            ax[pmra_ax_i].scatter(
                stream1_data['phi1'], stream1_data['PMRA'], marker=10, s=40,
                c=stream1_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            ax[pmra_ax_i].scatter(
                stream2_data['phi1'], stream2_data['PMRA'], marker=11, s=40,
                c=stream2_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            ax[pmra_ax_i].errorbar(
                stream1_data['phi1'], stream1_data['PMRA'],
                yerr=stream1_data['PMRA_ERROR'],
                capsize=0, elinewidth=0.75, ecolor='tab:blue', ms=6, fmt='none', zorder=0
            )
            ax[pmra_ax_i].errorbar(
                stream2_data['phi1'], stream2_data['PMRA'],
                yerr=stream2_data['PMRA_ERROR'],
                capsize=0, elinewidth=0.75, ecolor='tab:orange', ms=6, fmt='none', zorder=0
            )

        # Panel 3: PMDEC vs phi1 (or residual)
        if use_residual:
            ax[pmdec_ax_i].scatter(
                stream1_data['phi1'], s1_pmdec, marker=10, s=40,
                c=stream1_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            ax[pmdec_ax_i].scatter(
                stream2_data['phi1'], s2_pmdec, marker=11, s=40,
                c=stream2_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            ax[pmdec_ax_i].errorbar(
                stream1_data['phi1'], s1_pmdec,
                yerr=stream1_data['PMDEC_ERROR'],
                capsize=0, elinewidth=0.75, ecolor='tab:blue', ms=6, fmt='none', zorder=0
            )
            ax[pmdec_ax_i].errorbar(
                stream2_data['phi1'], s2_pmdec,
                yerr=stream2_data['PMDEC_ERROR'],
                capsize=0, elinewidth=0.75, ecolor='tab:orange', ms=6, fmt='none', zorder=0
            )
        else:
            ax[pmdec_ax_i].scatter(
                stream1_data['phi1'], stream1_data['PMDEC'], marker=10, s=40,
                c=stream1_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            ax[pmdec_ax_i].scatter(
                stream2_data['phi1'], stream2_data['PMDEC'], marker=11, s=40,
                c=stream2_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            ax[pmdec_ax_i].errorbar(
                stream1_data['phi1'], stream1_data['PMDEC'],
                yerr=stream1_data['PMDEC_ERROR'],
                capsize=0, elinewidth=0.75, ecolor='tab:blue', ms=6, fmt='none', zorder=0
            )
            ax[pmdec_ax_i].errorbar(
                stream2_data['phi1'], stream2_data['PMDEC'],
                yerr=stream2_data['PMDEC_ERROR'],
                capsize=0, elinewidth=0.75, ecolor='tab:orange', ms=6, fmt='none', zorder=0
            )
        # Optional Panel: Distance modulus vs phi1
        if include_dm_panel and dm_ax_i is not None:
            ax[dm_ax_i].scatter(
                stream1_data['phi1'], dm1, marker=10, s=40,
                c=stream1_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            ax[dm_ax_i].scatter(
                stream2_data['phi1'], dm2, marker=11, s=40,
                c=stream2_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
            )
            # Error bars can be symmetric value or tuple (minus, plus)
            try:
                ax[dm_ax_i].errorbar(
                    stream1_data['phi1'], dm1, yerr=dm1_err,
                    capsize=0, elinewidth=0.75, ecolor='tab:blue', ms=6, fmt='none', zorder=0
                )
            except Exception:
                pass
            try:
                ax[dm_ax_i].errorbar(
                    stream2_data['phi1'], dm2, yerr=dm2_err,
                    capsize=0, elinewidth=0.75, ecolor='tab:orange', ms=6, fmt='none', zorder=0
                )
            except Exception:
                pass
        
        # Final Panel: FEH vs phi1
        ax[feh_ax_i].scatter(
            stream1_data['phi1'], stream1_data['FEH'], marker=10, s=40,
            c=stream1_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
        )
        ax[feh_ax_i].scatter(
            stream2_data['phi1'], stream2_data['FEH'], marker=11, s=40,
            c=stream2_data['stream_prob'], cmap=cmap, alpha=0.8, zorder=1
        )
        ax[feh_ax_i].errorbar(
            stream1_data['phi1'], stream1_data['FEH'],
            yerr=stream1_data['FEH_ERR'],
            capsize=0, elinewidth=0.75, ecolor='tab:blue', ms=6, fmt='none', zorder=0
        )
        ax[feh_ax_i].errorbar(
            stream2_data['phi1'], stream2_data['FEH'],
            yerr=stream2_data['FEH_ERR'],
            capsize=0, elinewidth=0.75, ecolor='tab:orange', ms=6, fmt='none', zorder=0
        )
        
        # Add spline tracks for both streams if available (for visual reference)
        phi1_min = min(np.min(stream1_data['phi1']), np.min(stream2_data['phi1']))
        phi1_max = max(np.max(stream1_data['phi1']), np.max(stream2_data['phi1']))
        x_arr = np.linspace(phi1_min - 1, phi1_max + 1, 100)

        # show cbar
        norm = Normalize(vmin=0.5, vmax=1)
        cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.77])  # Adjust the position of the colorbar
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
        cbar.set_label('Stream Probability', fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        # Stream 1 splines (blue)
        spd1 = self.stream1.spline_points_dict
        nd1 = self.stream1.nested_dict
        spd2 = self.stream2.spline_points_dict
        nd2 = self.stream2.nested_dict

        # collect legend handles for panel 2 (index 1)
        spline_leg_handles = []

        if hasattr(self.stream1, 'spline_points_dict') and hasattr(self.stream1, 'nested_dict'):
            if use_residual and residual == 1:
                # reference is stream1: show spline points with errorbars centered at 0
                for panel_idx, med_idx in zip([1, 2, 3], [self.vgsr_idx, self.pmra_idx, self.pmdec_idx]):
                    _, em, ep = fetch_exp(nd1, med_idx)
                    try:
                        yerr = (em, ep) if (em is not None or ep is not None) else None
                        ax[panel_idx].errorbar(
                            spd1['phi1_spline_points'], np.zeros_like(spd1['phi1_spline_points']),
                            yerr=yerr, capsize=3, elinewidth=1, ms=6, fmt='o', mfc='tab:blue', mec='k', zorder=3, alpha=0.9
                        )
                    except Exception:
                        ax[panel_idx].plot(
                            spd1['phi1_spline_points'], np.zeros_like(spd1['phi1_spline_points']), 'o', mfc='tab:blue', mec='k', zorder=3, alpha=0.9
                        )
            elif not use_residual:
                # normal plotting of stream1 spline when not reference residual (or residual==0)
                h1, = ax[vgsr_ax_i].plot(
                    x_arr,
                    apply_spline(x_arr, spd1['phi1_spline_points'], nd1['meds'][self.vgsr_idx], spd1['spline_k']),
                    color='tab:blue', lw=2, zorder=2, ls='--', alpha=0.8, label=f"{stream1_label} spline"
                )
                spline_leg_handles.append(h1)
                ax[pmra_ax_i].plot(
                    x_arr, apply_spline(x_arr, spd1['phi1_spline_points'], nd1['meds'][self.pmra_idx], spd1['spline_k']),
                    color='tab:blue', lw=2, zorder=2, ls='--', alpha=0.8
                )
                ax[pmdec_ax_i].plot(
                    x_arr, apply_spline(x_arr, spd1['phi1_spline_points'], nd1['meds'][self.pmdec_idx], spd1['spline_k']),
                    color='tab:blue', lw=2, zorder=2, ls='--', alpha=0.8
                )

            # spline points & errorbars for stream1 (if not reference plotted as zeros)
            for panel_idx, med_idx in zip([vgsr_ax_i, pmra_ax_i, pmdec_ax_i], [self.vgsr_idx, self.pmra_idx, self.pmdec_idx]):
                med_vals, em, ep = fetch_exp(nd1, med_idx)
                try:
                    if residual == 1 and use_residual:
                        # when ref is stream1 we already plotted zeros+errors; skip plotting med-centered points to avoid confusion
                        continue
                    if med_vals is not None:
                        ax[panel_idx].errorbar(
                            spd1['phi1_spline_points'], med_vals,
                            yerr=(em, ep) if (em is not None or ep is not None) else None,
                            capsize=3, elinewidth=1, ms=6, fmt='o', mfc='tab:blue', mec='k', zorder=3, alpha=0.8
                        )
                except Exception:
                    pass

        # Stream 2 splines (orange)
        if hasattr(self.stream2, 'spline_points_dict') and hasattr(self.stream2, 'nested_dict'):
            spd2 = self.stream2.spline_points_dict
            nd2 = self.stream2.nested_dict

            if residual == 2 and use_residual:
                # reference is stream2: show spline points with errorbars centered at 0 for panels 1-3
                for panel_idx, med_idx in zip([1, 2, 3], [self.vgsr_idx, self.pmra_idx, self.pmdec_idx]):
                    _, em, ep = fetch_exp(nd2, med_idx)
                    try:
                        yerr = (em, ep) if (em is not None or ep is not None) else None
                        ax[panel_idx].errorbar(
                            spd2['phi1_spline_points'], np.zeros_like(spd2['phi1_spline_points']),
                            yerr=yerr, capsize=3, elinewidth=1, ms=6, fmt='o', mfc='tab:orange', mec='k', zorder=3, alpha=0.9
                        )
                    except Exception:
                        ax[panel_idx].plot(
                            spd2['phi1_spline_points'], np.zeros_like(spd2['phi1_spline_points']), 'o', mfc='tab:orange', mec='k', zorder=3, alpha=0.9
                        )
            elif not use_residual:
                # normal plotting of stream2 spline when not reference residual (or residual==0)
                h2, = ax[vgsr_ax_i].plot(
                    x_arr,
                    apply_spline(x_arr, spd2['phi1_spline_points'], nd2['meds'][self.vgsr_idx], spd2['spline_k']),
                    color='tab:orange', lw=2, zorder=2, ls='--', alpha=0.8, label=f"{stream2_label} spline"
                )
                spline_leg_handles.append(h2)
                ax[pmra_ax_i].plot(
                    x_arr, apply_spline(x_arr, spd2['phi1_spline_points'], nd2['meds'][self.pmra_idx], spd2['spline_k']),
                    color='tab:orange', lw=2, zorder=2, ls='--', alpha=0.8
                )
                ax[pmdec_ax_i].plot(
                    x_arr, apply_spline(x_arr, spd2['phi1_spline_points'], nd2['meds'][self.pmdec_idx], spd2['spline_k']),
                    color='tab:orange', lw=2, zorder=2, ls='--', alpha=0.8
                )
                # also plot stream2 spline points with error bars (medians) in non-residual mode
                for panel_idx, med_idx in zip([vgsr_ax_i, pmra_ax_i, pmdec_ax_i], [self.vgsr_idx, self.pmra_idx, self.pmdec_idx]):
                    med_vals, em, ep = fetch_exp(nd2, med_idx)
                    try:
                        if med_vals is not None:
                            ax[panel_idx].errorbar(
                                spd2['phi1_spline_points'], med_vals,
                                yerr=(em, ep) if (em is not None or ep is not None) else None,
                                capsize=3, elinewidth=1, ms=6, fmt='o', mfc='tab:orange', mec='k', zorder=3, alpha=0.9
                            )
                    except Exception:
                        pass

            # If reference is stream1 and we're showing residuals, plot stream2 spline as residual to reference
            if residual == 1 and use_residual:
                try:
                    y_vgsr = apply_spline(x_arr, spd2['phi1_spline_points'], nd2['meds'][self.vgsr_idx], spd2['spline_k']) - apply_spline(x_arr, spd1['phi1_spline_points'], nd1['meds'][self.vgsr_idx], spd1['spline_k'])
                    y_pmra = apply_spline(x_arr, spd2['phi1_spline_points'], nd2['meds'][self.pmra_idx], spd2['spline_k']) - apply_spline(x_arr, spd1['phi1_spline_points'], nd1['meds'][self.pmra_idx], spd1['spline_k'])
                    y_pmdec = apply_spline(x_arr, spd2['phi1_spline_points'], nd2['meds'][self.pmdec_idx], spd2['spline_k']) - apply_spline(x_arr, spd1['phi1_spline_points'], nd1['meds'][self.pmdec_idx], spd1['spline_k'])
                    h_diff21, = ax[vgsr_ax_i].plot(x_arr, y_vgsr, color='tab:orange', lw=2, zorder=2, ls='--', alpha=0.9, label=f"{stream2_label} - {stream1_label}")
                    spline_leg_handles.append(h_diff21)
                    ax[pmra_ax_i].plot(x_arr, y_pmra, color='tab:orange', lw=2, zorder=2, ls='--', alpha=0.9)
                    ax[pmdec_ax_i].plot(x_arr, y_pmdec, color='tab:orange', lw=2, zorder=2, ls='--', alpha=0.9)
                except Exception:
                    pass
                # plot the spline points for stream2 as residuals (meds - ref_evaluated_at_spd2_phi)
                for panel_idx, med_idx in zip([vgsr_ax_i, pmra_ax_i, pmdec_ax_i], [self.vgsr_idx, self.pmra_idx, self.pmdec_idx]):
                    med_vals, em, ep = fetch_exp(nd2, med_idx)
                    try:
                        if med_vals is None:
                            continue
                        ref_at_points = apply_spline(spd2['phi1_spline_points'], spd1['phi1_spline_points'], nd1['meds'][med_idx], spd1['spline_k'])
                        ypts = med_vals - ref_at_points
                        yerr = (em, ep) if (em is not None or ep is not None) else None
                        ax[panel_idx].errorbar(spd2['phi1_spline_points'], ypts, yerr=yerr, capsize=3, elinewidth=1, ms=6, fmt='o', mfc='tab:orange', mec='k', zorder=3, alpha=0.9)
                    except Exception:
                        pass

            # If reference is stream2 and we're showing residuals, plot stream1 spline as residual to stream2
            if residual == 2 and use_residual:
                try:
                    y_vgsr = apply_spline(x_arr, spd1['phi1_spline_points'], nd1['meds'][self.vgsr_idx], spd1['spline_k']) - apply_spline(x_arr, spd2['phi1_spline_points'], nd2['meds'][self.vgsr_idx], spd2['spline_k'])
                    y_pmra = apply_spline(x_arr, spd1['phi1_spline_points'], nd1['meds'][self.pmra_idx], spd1['spline_k']) - apply_spline(x_arr, spd2['phi1_spline_points'], nd2['meds'][self.pmra_idx], spd2['spline_k'])
                    y_pmdec = apply_spline(x_arr, spd1['phi1_spline_points'], nd1['meds'][self.pmdec_idx], spd1['spline_k']) - apply_spline(x_arr, spd2['phi1_spline_points'], nd2['meds'][self.pmdec_idx], spd2['spline_k'])
                    h_diff12, = ax[vgsr_ax_i].plot(x_arr, y_vgsr, color='tab:blue', lw=2, zorder=2, ls='--', alpha=0.9, label=f"{stream1_label} - {stream2_label}")
                    spline_leg_handles.append(h_diff12)
                    ax[pmra_ax_i].plot(x_arr, y_pmra, color='tab:blue', lw=2, zorder=2, ls='--', alpha=0.9)
                    ax[pmdec_ax_i].plot(x_arr, y_pmdec, color='tab:blue', lw=2, zorder=2, ls='--', alpha=0.9)
                except Exception:
                    pass
                # plot the spline points for stream1 as residuals (meds - ref_evaluated_at_spd1_phi)
                for panel_idx, med_idx in zip([vgsr_ax_i, pmra_ax_i, pmdec_ax_i], [self.vgsr_idx, self.pmra_idx, self.pmdec_idx]):
                    med_vals, em, ep = fetch_exp(nd1, med_idx)
                    try:
                        if med_vals is None:
                            continue
                        ref_at_points = apply_spline(spd1['phi1_spline_points'], spd2['phi1_spline_points'], nd2['meds'][med_idx], spd2['spline_k'])
                        ypts = med_vals - ref_at_points
                        yerr = (em, ep) if (em is not None or ep is not None) else None
                        ax[panel_idx].errorbar(spd1['phi1_spline_points'], ypts, yerr=yerr, capsize=3, elinewidth=1, ms=6, fmt='o', mfc='tab:blue', mec='k', zorder=3, alpha=0.9)
                    except Exception:
                        pass

        # Always show FEH spline (non-residual) for both streams if available
        # Stream1 FEH spline + spline points
        if hasattr(self.stream1, 'spline_points_dict') and hasattr(self.stream1, 'nested_dict'):
            spd1 = self.stream1.spline_points_dict
            nd1 = self.stream1.nested_dict
            try:
                ax[feh_ax_i].plot(x_arr, apply_spline(x_arr, spd1['phi1_spline_points'], nd1['meds'][3], spd1['spline_k']),
                       color='tab:blue', lw=2, zorder=2, ls='--', alpha=0.9)
                med_vals, em, ep = fetch_exp(nd1, 3)
                if med_vals is not None:
                    ax[feh_ax_i].errorbar(spd1['phi1_spline_points'], med_vals,
                           yerr=(em, ep) if (em is not None or ep is not None) else None,
                           capsize=3, elinewidth=1, ms=6, fmt='o', mfc='tab:blue', mec='k', zorder=3, alpha=0.9)
            except Exception:
                pass

        # Stream2 FEH spline + spline points
        if hasattr(self.stream2, 'spline_points_dict') and hasattr(self.stream2, 'nested_dict'):
            spd2 = self.stream2.spline_points_dict
            nd2 = self.stream2.nested_dict
            try:
                ax[feh_ax_i].plot(x_arr, apply_spline(x_arr, spd2['phi1_spline_points'], nd2['meds'][3], spd2['spline_k']),
                       color='tab:orange', lw=2, zorder=2, ls='--', alpha=0.9)
                med_vals, em, ep = fetch_exp(nd2, 3)
                if med_vals is not None:
                    ax[feh_ax_i].errorbar(spd2['phi1_spline_points'], med_vals,
                           yerr=(em, ep) if (em is not None or ep is not None) else None,
                           capsize=3, elinewidth=1, ms=6, fmt='o', mfc='tab:orange', mec='k', zorder=3, alpha=0.9)
            except Exception:
                pass

        # Draw a dashed zero line for residual plots (label only once on panel 2 for legend)
        if use_residual:
            zero_color = 'tab:blue' if residual == 1 else 'tab:orange'
            zero_label = (f"Ref: {stream1_label} spline" if residual == 1 else f"Ref: {stream2_label} spline")
            # label on panel 2
            h0 = ax[vgsr_ax_i].axhline(0, color=zero_color, lw=1, zorder=2, alpha=0.9, ls='--', label=zero_label)
            spline_leg_handles.append(h0)
            # repeat on other residual panels (no label)
            for panel_idx in [pmra_ax_i, pmdec_ax_i]:
                ax[panel_idx].axhline(0, color=zero_color, lw=1, zorder=2, alpha=0.9, ls='--')

        # Set labels (change to Delta labels when residuals applied)
        ax[phi2_ax_i].set_ylabel(r'$\phi_2$ (deg)', fontsize=14)
        if use_residual:
            ax[vgsr_ax_i].set_ylabel(r'$\Delta V_{GSR}$ (km/s)', fontsize=14)
            ax[pmra_ax_i].set_ylabel(r'$\Delta \mu_{\alpha}$ (mas/yr)', fontsize=14)
            ax[pmdec_ax_i].set_ylabel(r'$\Delta \mu_{\delta}$ (mas/yr)', fontsize=14)
        else:
            ax[vgsr_ax_i].set_ylabel(r'$V_{GSR}$ (km/s)', fontsize=14)
            ax[pmra_ax_i].set_ylabel(r'$\mu_{\alpha}$ (mas/yr)', fontsize=14)
            ax[pmdec_ax_i].set_ylabel(r'$\mu_{\delta}$ (mas/yr)', fontsize=14)
        if include_dm_panel and dm_ax_i is not None:
            ax[dm_ax_i].set_ylabel('Distance modulus (mag)', fontsize=14)
        ax[feh_ax_i].set_ylabel(r'[Fe/H]', fontsize=14)
        ax[feh_ax_i].set_xlabel(r'$\phi_1$ (deg)', fontsize=14)
        
        # Set title and legends
        if title:
            ax[phi2_ax_i].set_title(title, fontsize=16)
        ax[phi2_ax_i].legend()
        # add legend for splines on the second panel if any handles were collected
        if len(spline_leg_handles) > 0:
            ax[vgsr_ax_i].legend(handles=spline_leg_handles, fontsize=9, frameon=False)
        
        # Set xlim based on combined phi1 values
        phi1_min = min(np.min(stream1_data['phi1']), np.min(stream2_data['phi1']))
        phi1_max = max(np.max(stream1_data['phi1']), np.max(stream2_data['phi1']))
        ax[phi2_ax_i].set_xlim(phi1_min - 2, phi1_max + 2)
        
        # Set ylims based on combined stream data y values (for residuals use combined residual arrays if available)
        phi2_combined = np.concatenate([stream1_data['phi2'], stream2_data['phi2']])
        if use_residual:
            vgsr_combined = np.concatenate([s1_vgsr, s2_vgsr])
            pmra_combined = np.concatenate([s1_pmra, s2_pmra])
            pmdec_combined = np.concatenate([s1_pmdec, s2_pmdec])
        else:
            vgsr_combined = np.concatenate([stream1_data['VGSR'], stream2_data['VGSR']])
            pmra_combined = np.concatenate([stream1_data['PMRA'], stream2_data['PMRA']])
            pmdec_combined = np.concatenate([stream1_data['PMDEC'], stream2_data['PMDEC']])
        feh_combined = np.concatenate([stream1_data['FEH'], stream2_data['FEH']])
        
        ax[phi2_ax_i].set_ylim(np.min(phi2_combined) - 2, np.max(phi2_combined) + 2)
        ax[vgsr_ax_i].set_ylim(np.min(vgsr_combined) - 10, np.max(vgsr_combined) + 10)
        ax[pmra_ax_i].set_ylim(np.min(pmra_combined) - 1, np.max(pmra_combined) + 1)
        ax[pmdec_ax_i].set_ylim(np.min(pmdec_combined) - 1, np.max(pmdec_combined) + 1)
        ax[feh_ax_i].set_ylim(np.min(feh_combined) - 0.2, np.max(feh_combined) + 0.2)

        # Set scale and nice limits for the DM panel
        if include_dm_panel and dm_ax_i is not None:
            # No log scale for magnitudes; set margins based on data ranges if available
            try:
                dm_combined = np.concatenate([np.atleast_1d(dm1), np.atleast_1d(dm2)])
                pad = 0.2 if np.isfinite(dm_combined).all() else 0.0
                ymin, ymax = np.nanmin(dm_combined) - pad, np.nanmax(dm_combined) + pad
                if np.isfinite(ymin) and np.isfinite(ymax):
                    ax[dm_ax_i].set_ylim(ymin, ymax)
            except Exception:
                pass
        
        # Apply formatting to all axes
        for a in ax:
            plot_form(a)
        
        # Add background if requested (would need all_memberships data)
        if addBackground:
            print("Background plotting not implemented for stream comparison - requires all_memberships data")
        
        # Set rasterization for better PDF output
        for ax_curr in plt.gcf().get_axes():
            for artist in ax_curr.get_children():
                artist.set_rasterized(True)
        
        # Save figure if requested
        if save_fig:
            if fig_path is None:
                fig_path = 'figures_draft/postmcmc_6panel_comparison.pdf'
            plt.savefig(fig_path, bbox_inches='tight', dpi=600)
        
        if return_axes:
            return fig, ax
        else:
            plt.show()