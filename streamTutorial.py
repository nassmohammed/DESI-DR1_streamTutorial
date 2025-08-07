import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import pandas as pd
import healpy as hp
import scipy as sp
import scipy.stats as stats
from astropy.io import fits
from astropy import table
import astropy.coordinates as coord
from astropy.coordinates.matrix_utilities import rotation_matrix
import astropy.units as u
import matplotlib
import importlib
import stream_functions as stream_funcs
import emcee
import corner
from astroquery.gaia import Gaia
from astropy.table import Table, join
from collections import OrderedDict
import time
from scipy.interpolate import interp1d
import os
import feh_correct
import warnings
from astropy.utils.exceptions import AstropyDeprecationWarning
import copy
import multiprocessing
# Suppress specific Astropy deprecation warnings
warnings.filterwarnings("ignore", category=AstropyDeprecationWarning, module='gala.dynamics.core')


class Data:
    def __init__(self, desi_path, sf_path='/raid/catalogs/streamfinder_gaiadr3.fits'):
        self.desi_path = desi_path
        self.sf_path = sf_path

        # These are the columns from the DESI data that we want to import
        desired_columns = [
        'VRAD', 'VRAD_ERR', 'RVS_WARN', 'TEFF', 'LOGG', ## TEFF and LOGG needed for FeH correction
        'RR_SPECTYPE', 
        'TARGET_RA', 'TARGET_DEC', 'FEH', 'FEH_ERR', 'EBV', 'FLUX_G', 'FLUX_R', 'FLUX_Z',
        'TARGETID', 'PRIMARY', 'PHOT_BP_MEAN_FLUX', 'PHOT_RP_MEAN_FLUX',
        'SOURCE_ID', 'PMRA', 'PMRA_ERROR', 'PMDEC', 'PMDEC_ERROR', 'PARALLAX', 'PARALLAX_ERROR', 'PMRA_PMDEC_CORR'
        ]
        desi_hdu_indices = [1,3,4,5]
        self.desi_data = stream_funcs.load_fits_columns(desi_path, desired_columns, desi_hdu_indices)
        self.desi_data.label='DESI'
        # Drop the rows with NaN values in all columns
        print(f"Length of DESI Data before Cuts: {len(self.desi_data)}")
        self.desi_data = stream_funcs.dropna_Table(self.desi_data, columns = desired_columns)
        self.desi_data = self.desi_data[(self.desi_data['RVS_WARN'] == 0) & (self.desi_data['RR_SPECTYPE'] == 'STAR') & (self.desi_data['PRIMARY']) &\
        (self.desi_data['VRAD_ERR'] < 10) & (self.desi_data['FEH_ERR'] < 0.5)] 
        self.desi_data.remove_columns(['RVS_WARN', 'RR_SPECTYPE'])
        self.desi_data = self.desi_data.to_pandas()

        print(f"Length after NaN cut: {len(self.desi_data)}")

        # Applying additional errors in quadrature
        self.desi_data['VRAD_ERR'] = np.sqrt(self.desi_data['VRAD_ERR']**2 + 0.9**2) ### Turn into its own column
        self.desi_data['PMRA_ERROR'] = np.sqrt(self.desi_data['PMRA_ERROR']**2 + (np.sqrt(550)*0.001)**2) ### Turn into its own column
        self.desi_data['PMDEC_ERROR'] = np.sqrt(self.desi_data['PMDEC_ERROR']**2 + (np.sqrt(550)*0.001)**2) ### Turn into its own column
        self.desi_data['FEH_ERR'] = np.sqrt(self.desi_data['FEH_ERR']**2 + 0.01**2) ### Turn into its own column


        # Apply Metallicity correction to the DR1 Data (See Section 4.2 https://arxiv.org/pdf/2505.14787)
        print("Adding empirical FEH calibration (can find uncalibrated data in column['FEH_uncalib])")
        self.desi_data['FEH_uncalib'] = self.desi_data['FEH']
        self.desi_data['FEH'] = feh_correct.calibrate(self.desi_data['FEH'], self.desi_data['TEFF'], self.desi_data['LOGG'])
        
        # Switch to VGSR instead of VRAD
        self.desi_data['VGSR'] =  np.array(stream_funcs.vhel_to_vgsr(np.array(self.desi_data['TARGET_RA'])*u.deg, np.array(self.desi_data['TARGET_DEC'])*u.deg, np.array(self.desi_data['VRAD'])*u.km/u.s).value)

        # Lets load the STREAMFINDER data for Gaia DR3
        if sf_path:
            sf_data = table.Table.read(self.sf_path)
            self.sf_data = sf_data.to_pandas()
            self.sf_data['VGSR'] = np.array(stream_funcs.vhel_to_vgsr(np.array(self.sf_data['RAdeg'])*u.deg, np.array(self.sf_data['DEdeg'])*u.deg, np.array(self.sf_data['VHel'])*u.km/u.s).value)
        else: 
            print('No STREAMFINDER path given.')

        

    def select(self, mask_or_func):
        """
        Applies a mask and returns a new, filtered Data object.

        Args:
            mask_or_func (function or pd.Series): 
                - If a function, it must take a DataFrame and return a boolean Series.
                - If a Series, it must be a boolean mask with an index matching desi_data.

        Returns:
            Data: A new Data object containing only the filtered data.
        """
        new_data_object = copy.copy(self)
        original_df = new_data_object.desi_data

        if callable(mask_or_func):
            # It's a function, so call it to get the mask
            mask = mask_or_func(original_df)
        else:
            # Assume it's a pre-computed boolean series
            mask = mask_or_func
        
        new_data_object.desi_data = original_df[mask].copy()

        # Copy over stream-related attributes if they exist (for when this is called on stream data)
        stream_attrs = ['SoI_streamfinder', 'frame', 'SoI_galstream']
        for attr in stream_attrs:
            if hasattr(self, attr):
                setattr(new_data_object, attr, getattr(self, attr))
        
        return new_data_object

    def sfTable(self):
        ''''
        Not working, ask Joseph how he got his table
        
        '''
        if hasattr(self, 'sf_data') and hasattr(self, 'desi_data'):
            # Convert to pandas if needed
            sf_df = self.sf_data.to_pandas() if not isinstance(self.sf_data, pd.DataFrame) else self.sf_data
            desi_df = self.desi_data.to_pandas() if not isinstance(self.desi_data, pd.DataFrame) else self.desi_data

            # Inner join on Gaia == TARGETID to find common entries
            merged = pd.merge(sf_df, desi_df, left_on='Gaia', right_on='TARGETID', how='inner')

            # Count how many matched entries per Stream
            stream_counts = merged['Stream'].value_counts().reset_index()
            stream_counts.columns = ['Stream', 'Matched_Count']

            # Also count how many total entries per Stream in sf_data
            total_counts = sf_df['Stream'].value_counts().reset_index()
            total_counts.columns = ['Stream', 'SF_Count']

            # Merge both counts into one table
            stream_counts = pd.merge(total_counts, stream_counts, on='Stream', how='left')
            stream_counts['Matched_Count'] = stream_counts['Matched_Count'].fillna(0).astype(int)

            # Display the result
            print(stream_counts.to_string(index=False))

    def sfCrossMatch(self, isin=True):
        gaia_source_ids = self.SoI_streamfinder.columns[0]
        
        # Determine the attribute name to use
        if isin:
            # Store the original data for comparison if it exists
            # Priority: existing confirmed_sf_and_desi > confirmed_sf_and_desi_full > original_confirmed_sf_and_desi
            if hasattr(self, 'confirmed_sf_and_desi') and len(self.confirmed_sf_and_desi) > 0:
                self.old_base = self.confirmed_sf_and_desi.copy()
            elif hasattr(self, 'confirmed_sf_and_desi_full'):
                self.old_base = self.confirmed_sf_and_desi_full.copy()
            elif hasattr(self, 'original_confirmed_sf_and_desi'):
                self.old_base = self.original_confirmed_sf_and_desi.copy()
            
            base_name = 'confirmed_sf_and_desi'
            attr_name = base_name
            if hasattr(self, base_name):
                import string
                suffix = 'b'
                while hasattr(self ,f'{base_name}_{suffix}'):
                    suffix = chr(ord(suffix) + 1)
                attr_name = f'{base_name}_{suffix}'
        
            # Perform the merge and assign to the chosen attribute
            merged = pd.merge(
                self.SoI_streamfinder.drop_duplicates(subset=[gaia_source_ids]),
                self.desi_data.drop_duplicates(subset=['SOURCE_ID']),
                left_on=gaia_source_ids,
                right_on='SOURCE_ID',
                how='inner',
                suffixes=('_sf', '_desi')
            )
            merged.dropna(inplace=True)
            
            # Calculate phi1 and phi2 coordinates from TARGET_RA and TARGET_DEC (DESI coordinates)
            if len(merged) > 0:
                merged['phi1'], merged['phi2'] = stream_funcs.ra_dec_to_phi1_phi2(self.frame,np.array(merged['TARGET_RA'])*u.deg, np.array(merged['TARGET_DEC'])*u.deg)
            else:
                # If no matches, create empty phi1 and phi2 columns
                merged['phi1'] = pd.Series(dtype='float64')
                merged['phi2'] = pd.Series(dtype='float64')
            
            if 'VRAD' in merged.columns and len(merged) > 0 and merged['VRAD'].notnull().any():
                merged['VGSR'] = np.array(
                    stream_funcs.vhel_to_vgsr(
                        np.array(merged['TARGET_RA']) * u.deg,
                        np.array(merged['TARGET_DEC']) * u.deg,
                        np.array(merged['VRAD']) * u.km/u.s
                    ).value
                )
            else:
                # nans length of the dataframe or empty series if no data
                if len(merged) > 0:
                    merged['VGSR'] = np.nan * np.ones(len(merged))
                    print("No valid VRAD values found in 'merged'; skipping VGSR computation.")
                else:
                    merged['VGSR'] = pd.Series(dtype='float64')
            setattr(self, base_name, merged) # NOTE change base_name to attr_name if I want to not overwrite past confirmed_sf_and_desi
            if hasattr(self, 'old_base') and len(self.old_base) > 0:
                # Find stars that were in old_base but not in the new merged data
                # Use SOURCE_ID for comparison as it's the unique identifier
                old_source_ids = set(self.old_base['SOURCE_ID']) if 'SOURCE_ID' in self.old_base.columns else set()
                new_source_ids = set(merged['SOURCE_ID']) if 'SOURCE_ID' in merged.columns else set()
                cut_source_ids = old_source_ids - new_source_ids
                
                if cut_source_ids:
                    not_in_merged = self.old_base[self.old_base['SOURCE_ID'].isin(cut_source_ids)]
                    self.cut_confirmed_sf_and_desi = not_in_merged
                    print(f"Created cut_confirmed_sf_and_desi with {len(not_in_merged)} stars that were filtered out")
                else:
                    # If no stars were cut, create an empty DataFrame with same structure
                    self.cut_confirmed_sf_and_desi = self.old_base.iloc[0:0].copy()
                    print("No stars were cut - cut_confirmed_sf_and_desi is empty")
            else:
                # No old_base available - create empty DataFrame
                if hasattr(self, 'confirmed_sf_and_desi') and len(self.confirmed_sf_and_desi) > 0:
                    self.cut_confirmed_sf_and_desi = self.confirmed_sf_and_desi.iloc[0:0].copy()
                else:
                    self.cut_confirmed_sf_and_desi = pd.DataFrame()
                print("No original data available for comparison - cut_confirmed_sf_and_desi is empty")


            print(f"Number of stars in SF: {len(self.SoI_streamfinder)}, Number of DESI and SF stars: {len(merged)}")
            print(f"Saved merged DataFrame as self.data.{attr_name}")
        else:
            base_name = 'confirmed_sf_not_desi'
            attr_name = base_name
            if hasattr(self, base_name):
                import string
                suffix = 'b'
                while hasattr(self, f'{base_name}_{suffix}'):
                    suffix = chr(ord(suffix)+1)
                attr_name = f'{base_name}_{suffix}'
            unmatched = pd.merge(
            self.SoI_streamfinder.drop_duplicates(subset=[gaia_source_ids]),
            self.desi_data.drop_duplicates(subset=['SOURCE_ID']),
            left_on=gaia_source_ids,
            right_on='SOURCE_ID',
            how='outer',
            indicator=True,
            suffixes=('_sf', '_desi')
            )

            # Keep only the SoI_streamfinder rows that do not match any in desi_data
            only_in_SoI = unmatched[unmatched['_merge'] == 'left_only'].drop(columns=['_merge'])
            
            if len(only_in_SoI) > 0:
                only_in_SoI['phi1'], only_in_SoI['phi2'] = stream_funcs.ra_dec_to_phi1_phi2(self.frame,np.array(only_in_SoI['RAdeg'])*u.deg, np.array(only_in_SoI['DEdeg'])*u.deg)
                only_in_SoI['VGSR'] = np.array(stream_funcs.vhel_to_vgsr(np.array(only_in_SoI['RAdeg'])*u.deg, np.array(only_in_SoI['DEdeg'])*u.deg, np.array(only_in_SoI['VHel'])*u.km/u.s).value)
            else:
                # If no SF-only stars, create empty phi1, phi2, and VGSR columns
                only_in_SoI['phi1'] = pd.Series(dtype='float64')
                only_in_SoI['phi2'] = pd.Series(dtype='float64')
                only_in_SoI['VGSR'] = pd.Series(dtype='float64')
            
            setattr(self, base_name, only_in_SoI)
            print(f'Stars only in SF3: {len(only_in_SoI)}')

class Selection:
    """
    A class to manage and apply multiple selection criteria (masks) to a DataFrame.
    
    This class allows for the programmatic building of a complex filter by adding
    individual masks, which are then combined with a logical AND.
    """
    def __init__(self, data_frame):
        """
        Initializes the Selection object.

        Args:
            data_frame (pd.DataFrame): The pandas DataFrame to which the selections 
                                       will be applied.
        """
        if not isinstance(data_frame, pd.DataFrame):
            raise TypeError("Input 'data_frame' must be a pandas DataFrame.")
        
        self.df = data_frame
        self.masks = {} # A dictionary to store named mask functions
        print(f"Selection object created for DataFrame with {len(self.df)} rows.")

    def add_mask(self, name, mask_func):
        """
        Adds a new filtering criterion to the selection.

        Args:
            name (str): A descriptive name for the mask (e.g., 'metal_poor_cut').
            mask_func (function): A function that takes a DataFrame and returns a 
                                  boolean Series (the mask).
        """
        self.masks[name] = mask_func
        print(f"Mask added: '{name}'")

    def remove_mask(self, name):
        """Removes a mask by its name."""
        if name in self.masks:
            del self.masks[name]
            print(f"Mask removed: '{name}'")
        else:
            print(f"Warning: Mask '{name}' not found.")
            
    def list_masks(self):
        """Prints the names of all currently active masks."""
        if not self.masks:
            print("No masks are currently active.")
        else:
            print("Active masks:")
            for name in self.masks:
                print(f"- {name}")

    def get_final_mask(self):
        """
        Computes the final combined boolean mask.

        All individual masks are combined using a logical AND.

        Returns:
            pd.Series: A boolean Series representing the final combined mask.
        """
        if not self.masks:
            print("No masks to apply, returning an all-True mask.")
            return pd.Series([True] * len(self.df), index=self.df.index)

        # Start with a mask that is True for all entries
        final_mask = pd.Series(True, index=self.df.index)
        
        print("Combining masks...")
        for name, mask_func in self.masks.items():
            individual_mask = mask_func(self.df)
            final_mask &= individual_mask # Combine with logical AND
            print(f"...'{name}' selected {individual_mask.sum()} stars")

        print(f"Selection: {final_mask.sum()} / {len(self.df)} stars.")
        return final_mask
    
    def get_masks(self, mask_names):
        """
        Computes the final mask for a specific list of mask names.

        All individual masks are combined using a logical AND.

        Returns:
            pd.Series: A boolean Series representing the final combined mask for the specified names.
        """
        if not mask_names:
            print("No mask names provided, returning an all-True mask.")
            return pd.Series([True] * len(self.df), index=self.df.index)

        combined_mask = pd.Series(True, index=self.df.index)
        for name in mask_names:
            if name in self.masks:
                individual_mask = self.masks[name](self.df)
                combined_mask &= individual_mask  # Combine with logical AND
                print(f"...'{name}' selected {individual_mask.sum()} stars")
            else:
                print(f"Warning: Mask '{name}' not found. Skipping.")
        print(f"Selection for specified masks: {combined_mask.sum()} / {len(self.df)} stars.")

        return combined_mask


class stream:
    def __init__(self, data_object, streamName='Sylgr-I21', streamNo=42, frame=None):
        self.streamName = streamName
        self.streamNo = streamNo
        self.frame=frame

        # Store a reference to the data object instead of re-running the init
        self.data = data_object

        # Now, access the dataframes through the passed object
        # e.g., self.data.sf_data instead of self.sf_data
        self.data.SoI_streamfinder = self.data.sf_data[self.data.sf_data['Stream'] == self.streamNo]

        print('Importing galstreams module...')
        import galstreams
        mwsts = galstreams.MWStreams(verbose=False, implement_Off=True)
        self.data.SoI_galstream = mwsts.get(streamName, None)
        if (self.data.SoI_galstream is not None):
            self.min_dist = np.min(self.data.SoI_galstream.track.distance.value)
            print(self.min_dist)
        else:
            print('No galstream track available for this stream.')
        if (self.data.SoI_galstream is not None):
            self.frame = self.data.SoI_galstream.stream_frame
            self.data.frame = self.data.SoI_galstream.stream_frame
            self.data.SoI_galstream.gal_phi1 = self.data.SoI_galstream.track.transform_to(self.frame).phi1.deg
            self.data.SoI_galstream.gal_phi2 = self.data.SoI_galstream.track.transform_to(self.frame).phi2.deg
            
        print('Creating combined DataFrame of SF and DESI')
        # Access desi_data through self.data
        self.data.sfCrossMatch() #saved as confirmed_sf_and_desi
        self.data.sfCrossMatch(isin=False) #creates DF of stars not in DESI



        self.data.desi_data['phi1'], self.data.desi_data['phi2'] = stream_funcs.ra_dec_to_phi1_phi2(self.frame, np.array(self.data.desi_data['TARGET_RA'])*u.deg, np.array(self.data.desi_data['TARGET_DEC'])*u.deg)

        self.data.SoI_streamfinder['phi1'], self.data.SoI_streamfinder['phi2'] = stream_funcs.ra_dec_to_phi1_phi2(self.frame,np.array(self.data.SoI_streamfinder['RAdeg'])*u.deg, np.array(self.data.SoI_streamfinder['DEdeg'])*u.deg)

        self.data.confirmed_sf_and_desi['phi1'], self.data.confirmed_sf_and_desi['phi2'] = stream_funcs.ra_dec_to_phi1_phi2(self.frame,np.array(self.data.confirmed_sf_and_desi['TARGET_RA'])*u.deg, np.array(self.data.confirmed_sf_and_desi['TARGET_DEC'])*u.deg)

        self.data.confirmed_sf_not_desi['phi1'], self.data.confirmed_sf_not_desi['phi2'] = stream_funcs.ra_dec_to_phi1_phi2(self.frame,np.array(self.data.confirmed_sf_not_desi['RAdeg'])*u.deg, np.array(self.data.confirmed_sf_not_desi['DEdeg'])*u.deg)
        
        # convert sf from VHel to VGSR
        self.data.confirmed_sf_and_desi['VGSR'] = np.array(stream_funcs.vhel_to_vgsr(np.array(self.data.confirmed_sf_and_desi['TARGET_RA'])*u.deg, np.array(self.data.confirmed_sf_and_desi['TARGET_DEC'])*u.deg, np.array(self.data.confirmed_sf_and_desi['VRAD'])*u.km/u.s).value)
        self.data.confirmed_sf_not_desi['VGSR'] = np.array(stream_funcs.vhel_to_vgsr(np.array(self.data.confirmed_sf_not_desi['RAdeg'])*u.deg, np.array(self.data.confirmed_sf_not_desi['DEdeg'])*u.deg, np.array(self.data.confirmed_sf_not_desi['VHel'])*u.km/u.s).value)
    
    def mask_stream(self, mask_or_func):
        """
        Create a new stream object with filtered data and perform all necessary cross-matching.
        This replaces the 4-line pattern:
        - trimmed_desi = SoI.data.select(final_mask)
        - trimmed_stream = copy.copy(SoI)
        - trimmed_stream.data = trimmed_desi
        - trimmed_stream.data.sfCrossMatch(); trimmed_stream.data.sfCrossMatch(False)
        
        Args:
            mask_or_func: The mask or function to apply for filtering
        
        Returns:
            stream: A new stream object with filtered and cross-matched data
        """
        # Step 1: Apply the mask to the data
        trimmed_data = self.data.select(mask_or_func)
        
        # Step 2: Create a copy of the stream object
        trimmed_stream = copy.copy(self)
        
        # Step 3: Assign the filtered data
        trimmed_stream.data = trimmed_data
        
        # Step 4: Perform cross-matching
        trimmed_stream.data.sfCrossMatch()  # Creates confirmed_sf_and_desi
        trimmed_stream.data.sfCrossMatch(False)  # Creates confirmed_sf_not_desi
        
        # Step 5: Automatically compute VGSR for confirmed_sf_not_desi if it exists and has VHel data
        if hasattr(trimmed_stream.data, 'confirmed_sf_not_desi') and len(trimmed_stream.data.confirmed_sf_not_desi) > 0:
            if 'VHel' in trimmed_stream.data.confirmed_sf_not_desi.columns:
                # Copy VHel and set 0 values to np.nan
                vhel = np.array(trimmed_stream.data.confirmed_sf_not_desi['VHel'], dtype=float)
                vhel[vhel == 0] = np.nan
                
                # Only compute VGSR if we have valid VHel values
                if not np.all(np.isnan(vhel)):
                    # Compute VGSR using the stream_functions
                    trimmed_stream.data.confirmed_sf_not_desi['VGSR'] = stream_funcs.vhel_to_vgsr(
                        np.array(trimmed_stream.data.confirmed_sf_not_desi['RAdeg']) * u.deg,
                        np.array(trimmed_stream.data.confirmed_sf_not_desi['DEdeg']) * u.deg,
                        vhel * u.km/u.s
                    ).value
        
        return trimmed_stream

    def isochrone(self, metallicity, age, dotter_directory='./data/dotter/'):
        """
        Placeholder for isochrone fitting logic.
        """
        mass_fraction = 0.0181 * 10 ** metallicity

        dotter_mass_frac = np.array([
        0.00006, 0.00007, 0.00009, 0.00010, 0.00011, 0.00013, 0.00014, 0.00016,
        0.00017, 0.00019, 0.00021, 0.00024, 0.00028, 0.00032, 0.00037, 0.00042,
        0.00049, 0.00057, 0.00063, 0.00072, 0.00082, 0.00093, 0.00108, 0.00124,
        0.00144, 0.00166, 0.00189, 0.00213, 0.00242, 0.00276, 0.00316, 0.00363,
        0.00417
        ])
        dotter_mass_frac_str = [
        "0.00006", "0.00007", "0.00009", "0.00010", "0.00011", "0.00013", "0.00014", "0.00016",
        "0.00017", "0.00019", "0.00021", "0.00024", "0.00028", "0.00032", "0.00037", "0.00042",
        "0.00049", "0.00057", "0.00063", "0.00072", "0.00082", "0.00093", "0.00108", "0.00124",
        "0.00144", "0.00166", "0.00189", "0.00213", "0.00242", "0.00276", "0.00316", "0.00363",
        "0.00417"
    ]
        use_mass_frac = dotter_mass_frac_str[np.argmin(dotter_mass_frac - mass_fraction)]

        isochrone_path = dotter_directory + 'iso_a' + str(age) + '_z' + str(use_mass_frac) + '.dat'
        print(f'using {isochrone_path}')
        dotter_mp = np.loadtxt(isochrone_path)

        # Obtain the M_g and M_r color band data
        self.dotter_g_mp = dotter_mp[:,6]
        self.dotter_r_mp = dotter_mp[:,7]

        if np.round(self.min_dist,4) != 1:
            # interpolate distance
            interpolate_distances = interp1d(self.data.SoI_galstream.gal_phi1, self.data.SoI_galstream.track.distance.value*1000, kind='linear', fill_value='extrapolate')
            distance_sf = interpolate_distances(self.data.confirmed_sf_and_desi['phi1'])
            distance_desi = interpolate_distances(self.data.desi_data['phi1'])
            distance_cut_sf = interpolate_distances(self.data.cut_confirmed_sf_and_desi['phi1']) if hasattr(self.data, 'cut_confirmed_sf_and_desi') else None
            print('Using distance gradient')
        elif not self.data.confirmed_sf_and_desi.empty:
            distance_sf = 1/np.nanmean(self.data.confirmed_sf_and_desi['PARALLAX'])*1000
            distance_desi = distance_sf
            distance_cut_sf = 1/np.nanmean(self.data.cut_confirmed_sf_and_desi['PARALLAX'])*1000 if hasattr(self.data, 'cut_confirmed_sf_and_desi') else None
            print(f'set distance to {distance_sf} pc')
        else:
            print('No distance for the stream, go look in literature and set manually with self.min_dist = XX') #kpc)
        self.data.desi_colour_idx, self.data.desi_abs_mag, self.data.desi_r_mag = stream_funcs.get_colour_index_and_abs_mag(self.data.desi_data['EBV'], self.data.desi_data['FLUX_G'], self.data.desi_data['FLUX_R'], distance_desi)
        self.data.sf_colour_idx, self.data.sf_abs_mag, self.data.sf_r_mag = stream_funcs.get_colour_index_and_abs_mag(self.data.confirmed_sf_and_desi['EBV'], self.data.confirmed_sf_and_desi['FLUX_G'], self.data.confirmed_sf_and_desi['FLUX_R'], distance_sf)
        if hasattr(self.data, 'cut_confirmed_sf_and_desi'):
                self.data.cut_sf_colour_idx, self.data.cut_sf_abs_mag, self.data.cut_sf_r_mag = stream_funcs.get_colour_index_and_abs_mag(self.data.cut_confirmed_sf_and_desi['EBV'], self.data.cut_confirmed_sf_and_desi['FLUX_G'], self.data.cut_confirmed_sf_and_desi['FLUX_R'], distance_cut_sf)
        g_r_color_dif = self.dotter_g_mp - self.dotter_r_mp
        sorted_indices = np.argsort(self.dotter_r_mp)
        sorted_dotter_r_mp = self.dotter_r_mp[sorted_indices]
        g_r_color_dif = g_r_color_dif[sorted_indices]

        # Fit for the isochrone line
        self.isochrone_fit = sp.interpolate.UnivariateSpline(sorted_dotter_r_mp, g_r_color_dif, s=0)

#class Orbit: WIP

class StreamPlotter:
    """
    For really clean and easy plotting
    """
    def __init__(self, stream_or_mcmeta_object, save_dir='plots/'):
        """
        Initializes the plotter with a stream object or MCMeta object.
        
        Args:
            stream_or_mcmeta_object: Either a stream instance or MCMeta instance
            save_dir (str): Directory to save plots.
        """
        # Check if it's an MCMeta object or stream object
        if hasattr(stream_or_mcmeta_object, 'initial_params'):
            # It's an MCMeta object
            self.mcmeta = stream_or_mcmeta_object
            self.stream = stream_or_mcmeta_object.stream
            self.data = stream_or_mcmeta_object.stream.data
        else:
            # It's a stream object
            self.stream = stream_or_mcmeta_object
            self.data = stream_or_mcmeta_object.data
            self.mcmeta = None
            
        self.save_dir = save_dir
        # Create directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.plot_params = {
            'sf_in_desi': {
                'marker': 'd',
                's': 40,
                'color': 'green',
                'label': r'SF $\in$ DESI',
                'edgecolor': 'k',
                'zorder': 5
            },
            'sf_not_desi': {
                'marker': 'd',
                's': 30,
                'color': 'none',
                'alpha': 1,
                'edgecolor': 'k',
                'label': 'SF (not in DESI)',
                'zorder': 4
            },
            'sf_in_desi_notsel': {
                'marker': 'x',
                's': 70,
                'color': 'r',
                'label': r'SF $\in$ DESI, Cut',
                'edgecolor': 'k',
                'zorder': 5
            },
            'sf_not_desi_notsel': {
                'marker': 'd',
                's': 30,
                'color': 'none',
                'alpha': 1,
                'edgecolor': 'r',
                'label': 'SF (not in DESI), Cut',
                'zorder': 4
            },
            'desi_field': {
                'marker': '.',
                's': 1,
                'color': 'k',
                'alpha': 0.1,
                'label': 'DESI Field Stars',
                'zorder': 1
            },
            'galstream_track': {
                'color': 'y',
                'lw': 2,
                'alpha': 0.5,
                'label': 'galstream',
                'zorder': 2
            },
            'spline_track': {
                'color': 'b',
                'ls': '-.',
                'lw': 1,
                'label': 'Spline',
                'zorder': 3
            },
            'orbit_track': {
                'color': 'r',
                'ls': 'dotted',
                'lw': 1,
                'label': 'Best-fit Orbit',
                'zorder': 3
            },
            'background':{
                'color':'k',
                's':2,
                'label':'DESI',
                'alpha': 0.02,
                'zorder':0
            }
        }
    
    def on_sky(self, stream_frame=True, showStream=True, save=False, galstream=True, orbit=True, background=False):
        """
        Plots the stream on-sky either in RA, DEC or phi1, phi2
        """
        if stream_frame:
            col_x = 'phi1'
            col_x_ = 'phi1'
            label_x = r'$\phi_1$'
            col_y = 'phi2'
            col_y_ = 'phi2'
            label_y = r'$\phi_2$'
        else:
            col_x = 'TARGET_RA'
            col_x_ = 'RAdeg'
            col_y = 'TARGET_DEC'
            col_y_ = 'DEdeg'
            label_x = 'RA (deg)'
            label_y = 'DEC (deg)'

        fig, ax = plt.subplots(figsize=(10, 3))
        if showStream:
            ax.scatter(
                self.data.confirmed_sf_and_desi[col_x],
                self.data.confirmed_sf_and_desi[col_y],
                **self.plot_params['sf_in_desi']
            )
            ax.scatter(
                self.data.confirmed_sf_not_desi[col_x_],
                self.data.confirmed_sf_not_desi[col_y_],
                **self.plot_params['sf_not_desi']
            )
            if hasattr(self.data, 'cut_confirmed_sf_and_desi'):
                if showStream:
                    ax.scatter(
                        self.data.cut_confirmed_sf_and_desi[col_x],
                        self.data.cut_confirmed_sf_and_desi[col_y],
                        **self.plot_params['sf_in_desi_notsel']
                    )
            if stream_frame:
                if galstream:
                    ax.plot(
                        self.data.SoI_galstream.gal_phi1,
                        self.data.SoI_galstream.gal_phi2,
                        **self.plot_params['galstream_track']
                    )
            else:
                if galstream:
                    ax.plot(
                        self.data.SoI_galstream.track.ra,
                        self.data.SoI_galstream.track.dec,
                        **self.plot_params['galstream_track']
                    )

        if background:
            ax.scatter(
                self.data.desi_data[col_x],
                self.data.desi_data[col_y],
                **self.plot_params['background']
            )

        # Placeholder for orbit plotting logic
        # if hasattr(self.stream, orbit):
        #     ax.plot(
        #         self.orbit.<x>,
        #         self.orbit.<y>,
        #         **self.plot_params['orbit_track'])

        ax.legend(loc='upper left', ncol=4)
        ax.set_ylabel(label_y)
        ax.set_xlabel(label_x)
        stream_funcs.plot_form(ax)  # Make sure this is defined or imported

    def plx_cut(self, showStream=True, background=True, save=False, galstream=False):
        fig, ax = plt.subplots(figsize=(10, 5))
        col_x = 'TARGET_RA'
        col_x_ = 'RAdeg'
        label_x = 'RA (deg)'
        label_y = r'Parallax - 2* Paralalx Error'
        if background:
            ax.scatter(
                self.data.desi_data[col_x],
                self.data.desi_data['PARALLAX']-2*self.data.desi_data['PARALLAX_ERROR'],
                **self.plot_params['background']
            )
        if showStream:
            ax.scatter(
                self.data.confirmed_sf_and_desi[col_x],
                self.data.confirmed_sf_and_desi['PARALLAX']-2* self.data.confirmed_sf_and_desi['PARALLAX_ERROR'],
                **self.plot_params['sf_in_desi']
            )
            # ax.scatter(
            #     self.data.confirmed_sf_not_desi[col_x_],
            #     self.data.confirmed_sf_not_desi['plx']-2*np.nanmean(self.data.desi_data['PARALLAX_ERROR']), # NOTE, error is not given, may want to get from Gaia?
            #     **self.plot_params['sf_not_desi']
            # )

        # WIP, want to show if any stars are cut. Right now its failing for some reason.
        if hasattr(self.data, 'cut_confirmed_sf_and_desi'):
            if showStream:
                ax.scatter(
                    self.data.cut_confirmed_sf_and_desi[col_x],
                    self.data.cut_confirmed_sf_and_desi['PARALLAX']-2* self.data.cut_confirmed_sf_and_desi['PARALLAX_ERROR'],
                    **self.plot_params['sf_in_desi_notsel']
                )
        #         ax.scatter(
        #             self.data.notsel.confirmed_sf_not_desi[col_x_],
        #             self.data.notsel.confirmed_sf_not_desi['PARALLAX']-2* self.data.notsel.confirmed_sf_not_desi['PARALLAX_ERROR'],
        #             **self.plot_params['sf_not_desi_notsel']
        #         )

        ax.axhline(y=1/self.stream.min_dist, color='r', linestyle='--', label=f'1 / min_dist ({self.stream.min_dist:.2f})')
        ax.set_ylim(np.nanmin(self.data.SoI_streamfinder['plx'])-1,np.nanmax(self.data.SoI_streamfinder['plx']+1))
        ax.legend(loc='upper left', ncol=4)
        ax.set_ylabel(label_y)
        ax.set_xlabel(label_x)
        stream_funcs.plot_form(ax)  # Make sure this is defined or imported

    def kin_plot(self, showStream=True, show_sf_only=False, background=True, save=False, stream_frame=True):#, galstream=False):
        """
        Plots the stream kinematics either on-sky or stream_frame
        """
        if stream_frame:
            col_x = 'phi1'
            col_x_ = 'phi1'
            label_x = r'$\phi_1$'
            col_y = 'VGSR'
            col_y_ = 'VGSR'
            label_y = r'V_${GSR}$ (km/s)'
        else:
            col_x = 'TARGET_RA'
            col_x_ = 'RAdeg'
            col_y = 'VGSR'
            col_y_ = 'VGSR'
            label_x = 'RA (deg)'
            label_y = 'VGSR (km/s)'

        fig, ax = plt.subplots(3, 1, figsize=(10, 10))
        if showStream:
            ax[0].scatter(
                self.data.confirmed_sf_and_desi[col_x],
                self.data.confirmed_sf_and_desi[col_y],
                **self.plot_params['sf_in_desi']
            )
            ax[1].scatter(
                self.data.confirmed_sf_and_desi[col_x],
                self.data.confirmed_sf_and_desi['PMRA'],
                **self.plot_params['sf_in_desi']
            )
            ax[2].scatter(
                self.data.confirmed_sf_and_desi[col_x],
                self.data.confirmed_sf_and_desi['PMDEC'],
                **self.plot_params['sf_in_desi']
            )
            # WIP, option to show sf not in desi
            if hasattr(self.data, 'cut_confirmed_sf_and_desi'):
                if showStream:
                    ax[0].scatter(
                        self.data.cut_confirmed_sf_and_desi[col_x],
                        self.data.cut_confirmed_sf_and_desi[col_y_],
                        **self.plot_params['sf_in_desi_notsel']
                    )
                    ax[1].scatter(
                        self.data.cut_confirmed_sf_and_desi[col_x],
                        self.data.cut_confirmed_sf_and_desi['PMRA'],
                        **self.plot_params['sf_in_desi_notsel']
                    )
                    ax[2].scatter(
                        self.data.cut_confirmed_sf_and_desi[col_x],
                        self.data.cut_confirmed_sf_and_desi['PMDEC'],
                        **self.plot_params['sf_in_desi_notsel']
                    )
            # if stream_frame: WIP show galstream
            #     if galstream:
            #         ax[0].plot(
            #             self.data.SoI_galstream.gal_phi1,
            #             self.data.SoI_galstream.,
            #             **self.plot_params['galstream_track']
            #         )
            #         ax[0].plot(
            #             self.data.SoI_galstream.gal_phi1,
            #             self.data.SoI_galstream.gal_phi2,
            #             **self.plot_params['galstream_track']
            #         )
            # else:
            #     if galstream:
            #         ax[0].plot(
            #             self.data.SoI_galstream.track.ra,
            #             self.data.SoI_galstream.track.dec,
            #             **self.plot_params['galstream_track']
            #         )
        if show_sf_only:

            ax[0].scatter(
                self.data.confirmed_sf_not_desi[col_x_],
                self.data.confirmed_sf_not_desi[col_y_],
                **self.plot_params['sf_not_desi']
            )    
            ax[1].scatter(
                self.data.confirmed_sf_not_desi[col_x_],
                self.data.confirmed_sf_not_desi['pmRA'],
                **self.plot_params['sf_not_desi']
            )  
            ax[2].scatter(
                self.data.confirmed_sf_not_desi[col_x_],
                self.data.confirmed_sf_not_desi['pmDE'],
                **self.plot_params['sf_not_desi']
            )  
        if background:
            ax[0].scatter(
                self.data.desi_data[col_x],
                self.data.desi_data[col_y],
                **self.plot_params['background']
            )
            ax[1].scatter(
                self.data.desi_data[col_x],
                self.data.desi_data['PMRA'],
                **self.plot_params['background']
            )
            ax[2].scatter(
                self.data.desi_data[col_x],
                self.data.desi_data['PMDEC'],
                **self.plot_params['background']
            )

        # Placeholder for orbit plotting logic
        # if hasattr(self.stream, orbit):
        #     ax.plot(
        #         self.orbit.<x>,
        #         self.orbit.<y>,
        #         **self.plot_params['orbit_track'])

        if showStream:
            ax[0].set_ylim(np.nanmin(np.concatenate([self.data.confirmed_sf_and_desi[col_y_], self.data.cut_confirmed_sf_and_desi[col_y_]])) - 100,
                        np.nanmax(np.concatenate([self.data.confirmed_sf_and_desi[col_y_], self.data.cut_confirmed_sf_and_desi[col_y_]])) + 100)

            ax[1].set_ylim(np.nanmin(np.concatenate([self.data.confirmed_sf_and_desi['PMRA'], self.data.cut_confirmed_sf_and_desi['PMRA']])) - 7,
                        np.nanmax(np.concatenate([self.data.confirmed_sf_and_desi['PMRA'], self.data.cut_confirmed_sf_and_desi['PMRA']])) + 7)

            ax[2].set_ylim(np.nanmin(np.concatenate([self.data.confirmed_sf_and_desi['PMDEC'], self.data.cut_confirmed_sf_and_desi['PMDEC']])) - 7,
                        np.nanmax(np.concatenate([self.data.confirmed_sf_and_desi['PMDEC'], self.data.cut_confirmed_sf_and_desi['PMDEC']])) + 7)
        ax[0].legend(loc='upper left', ncol=4)
        ax[0].set_ylabel(label_y)
        ax[1].set_ylabel(r'$\mu_{\alpha}$ [mas/yr]')
        ax[2].set_ylabel(r'$\mu_{\delta}$ [mas/yr]')
        ax[-1].set_xlabel(label_x)
        stream_funcs.plot_form(ax[0])  
        stream_funcs.plot_form(ax[1]) 
        stream_funcs.plot_form(ax[2])

    def feh_plot(self, showStream=True, show_sf_only=False, background=True, save=False, stream_frame=True):
        """
        Plots the stream metallicity either on-sky or stream_frame
        """
        if stream_frame:
            col_x = 'phi1'
            col_x_ = 'phi1'
            label_x = r'$\phi_1$'
            col_y = 'FEH'
            col_y_ = 'FEH'
            label_y = r'[Fe/H]'
        else:
            col_x = 'TARGET_RA'
            col_x_ = 'RAdeg'
            col_y = 'FEH'
            col_y_ = 'FEH'
            label_x = 'RA (deg)'
            label_y = r'[Fe/H]'
        fig, ax = plt.subplots(figsize=(10, 5))
        if showStream:
            ax.scatter(
                self.data.confirmed_sf_and_desi[col_x],
                self.data.confirmed_sf_and_desi[col_y],
                **self.plot_params['sf_in_desi']
            )
            # WIP, option to show sf not in desi
            if hasattr(self.data, 'cut_confirmed_sf_and_desi'):
                if showStream:
                    ax.scatter(
                        self.data.cut_confirmed_sf_and_desi[col_x],
                        self.data.cut_confirmed_sf_and_desi[col_y_],
                        **self.plot_params['sf_in_desi_notsel']
                    )
        if show_sf_only:
            ax.scatter(
                self.data.confirmed_sf_not_desi[col_x_],
                self.data.confirmed_sf_not_desi[col_y_],
                **self.plot_params['sf_not_desi']
            )
        if background:
            ax.scatter(
                self.data.desi_data[col_x],
                self.data.desi_data[col_y],
                **self.plot_params['background']
            )
        
        ax.set_ylim(-4, 0.5)
        ax.legend(loc='upper left', ncol=4)
        ax.set_ylabel(label_y)
        ax.set_xlabel(label_x)
        stream_funcs.plot_form(ax) 

        return fig, ax

    def iso_plot(self, wiggle = 0.18, showStream=True, show_sf_only=False, background=True, save=False, absolute=True, BHB=True, bhb_wiggle=True):
        """
        Plotting the isochrone and stars
        """
        fig, ax = plt.subplots(figsize=(6, 7))
        if showStream:
            if absolute:
                ax.scatter(self.data.sf_colour_idx, self.data.sf_abs_mag,
                            **self.plot_params['sf_in_desi'])
            else:
                ax.scatter(self.data.sf_colour_idx, self.data.sf_r_mag,
                            **self.plot_params['sf_in_desi'])
        if background:
            if absolute:
                ax.scatter(self.data.desi_colour_idx, self.data.desi_abs_mag,
                            **self.plot_params['background'])
            else:
                ax.scatter(self.data.desi_colour_idx, self.data.desi_r_mag,
                            **self.plot_params['background'])
    
        ax.plot(self.stream.isochrone_fit(self.stream.dotter_r_mp), self.stream.dotter_r_mp,
                c='b', ls='-.')
        ax.plot(self.stream.isochrone_fit(self.stream.dotter_r_mp)+wiggle, self.stream.dotter_r_mp,
                c='b', ls='dotted', alpha=0.5, label='Colour wiggle')
        ax.plot(self.stream.isochrone_fit(self.stream.dotter_r_mp)-wiggle, self.stream.dotter_r_mp,
                c='b', ls='dotted', alpha=0.5)
        if hasattr(self.data, 'cut_confirmed_sf_and_desi'):
                if showStream:
                    ax.scatter(
                        self.data.cut_sf_colour_idx,
                        self.data.cut_sf_abs_mag if absolute else self.data.cut_sf_r_mag,
                        **self.plot_params['sf_in_desi_notsel']
                    )
        # Hard coded
        if BHB:

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
            ax.plot(des_m92_hb_g - des_m92_hb_r, des_m92_hb_r, c='b', alpha=1, ls='-.')
            if bhb_wiggle:
                bhb_color_wiggle = 0.4
                bhb_abs_mag_wiggle = 0.1
                ax.plot(des_m92_hb_g - des_m92_hb_r, des_m92_hb_r-bhb_color_wiggle, 'b:', alpha=0.5)
                ax.plot(des_m92_hb_g - des_m92_hb_r, des_m92_hb_r+bhb_color_wiggle, 'b:', alpha=0.5)
                ax.plot(des_m92_hb_g - des_m92_hb_r+bhb_abs_mag_wiggle, des_m92_hb_r, 'b:', alpha=0.5)
                ax.plot(des_m92_hb_g - des_m92_hb_r-bhb_abs_mag_wiggle, des_m92_hb_r, 'b:', alpha=0.5)
        # Hard coded
        
        ax.legend(loc='lower left')
        ax.set_xlabel('g-r',fontsize=15)
        ax.set_ylabel('$M_r$',fontsize=15)
        ax.set_xlim(-0.5, 1.2)
        ax.set_ylim(-1.5, 8)
        ax.invert_yaxis()
        stream_funcs.plot_form(ax)

    def sixD_plot(self, showStream=True, show_sf_only=False, background=True, save=False, stream_frame=True, galstream=False, show_cut=False, 
                  show_initial_splines=False, show_optimized_splines=False, show_mcmc_splines=False, show_sf_errors=True):
        """
        Plots the stream phi1 vs phi2, vgsr, pmra, pmdec, and feh
        
        Parameters:
        -----------
        show_cut : bool, optional
            Whether to show cut stars (red X markers). Default is True.
        show_initial_splines : bool, optional
            Whether to show initial guess splines in black. Default is False.
        show_optimized_splines : bool, optional
            Whether to show optimized splines in red. Default is False.
        show_mcmc_splines : bool, optional
            Whether to show MCMC results splines in blue. Default is False.
        show_sf_errors : bool, optional
            Whether to show error bars on StreamFinder stars. Default is True.
        """
        if stream_frame:
            col_x = 'phi1'
            col_x_ = 'phi1'
            label_x = r'$\phi_1$'
        else:
            col_x = 'TARGET_RA'
            col_x_ = 'RAdeg'
            label_x = 'RA (deg)'

        fig, ax = plt.subplots(5, 1, figsize=(10, 15))
        
        # Plot 1: phi2 vs phi1 (or DEC vs RA)
        if stream_frame:
            col_y0 = 'phi2'
            col_y0_ = 'phi2'
            label_y0 = r'$\phi_2$'
        else:
            col_y0 = 'TARGET_DEC'
            col_y0_ = 'DEdeg'
            label_y0 = 'DEC (deg)'
            
        if showStream:
            if show_sf_errors:
                # Plot with error bars - create compatible parameters for errorbar
                # Only use parameters that work with both errorbar and scatter
                errorbar_params = {
                    'color': self.plot_params['sf_in_desi'].get('color', 'green'),
                    'label': self.plot_params['sf_in_desi'].get('label', 'SF $\\in$ DESI'),
                    'zorder': self.plot_params['sf_in_desi'].get('zorder', 5),
                    'alpha': self.plot_params['sf_in_desi'].get('alpha', 1.0),
                    'markersize': 6,  # Use markersize instead of s
                    'markeredgecolor': self.plot_params['sf_in_desi'].get('edgecolor', 'k'),
                    'ecolor': self.plot_params['sf_in_desi'].get('edgecolor', 'k'),
                    'capsize': 2,
                }
                
                ax[0].errorbar(
                    self.data.confirmed_sf_and_desi[col_x],
                    self.data.confirmed_sf_and_desi[col_y0],
                    xerr=None, yerr=None,
                    fmt='d', **errorbar_params  # Use diamond marker like the original
                )
                # VGSR with error bars
                ax[1].errorbar(
                    self.data.confirmed_sf_and_desi[col_x],
                    self.data.confirmed_sf_and_desi['VGSR'],
                    xerr=None, yerr=self.data.confirmed_sf_and_desi['VRAD_ERR'],
                    fmt='d', **errorbar_params
                )
                # PMRA with error bars
                ax[2].errorbar(
                    self.data.confirmed_sf_and_desi[col_x],
                    self.data.confirmed_sf_and_desi['PMRA'],
                    xerr=None, yerr=self.data.confirmed_sf_and_desi['PMRA_ERROR'],
                    fmt='d', **errorbar_params
                )
                # PMDEC with error bars
                ax[3].errorbar(
                    self.data.confirmed_sf_and_desi[col_x],
                    self.data.confirmed_sf_and_desi['PMDEC'],
                    xerr=None, yerr=self.data.confirmed_sf_and_desi['PMDEC_ERROR'],
                    fmt='d', **errorbar_params
                )
                # FEH with error bars
                ax[4].errorbar(
                    self.data.confirmed_sf_and_desi[col_x],
                    self.data.confirmed_sf_and_desi['FEH'],
                    xerr=None, yerr=self.data.confirmed_sf_and_desi['FEH_ERR'],
                    fmt='d', **errorbar_params
                )
            else:
                # Plot without error bars (original behavior)
                ax[0].scatter(
                    self.data.confirmed_sf_and_desi[col_x],
                    self.data.confirmed_sf_and_desi[col_y0],
                    **self.plot_params['sf_in_desi']
                )
                ax[1].scatter(
                    self.data.confirmed_sf_and_desi[col_x],
                    self.data.confirmed_sf_and_desi['VGSR'],
                    **self.plot_params['sf_in_desi']
                )
                ax[2].scatter(
                    self.data.confirmed_sf_and_desi[col_x],
                    self.data.confirmed_sf_and_desi['PMRA'],
                    **self.plot_params['sf_in_desi']
                )
                ax[3].scatter(
                    self.data.confirmed_sf_and_desi[col_x],
                    self.data.confirmed_sf_and_desi['PMDEC'],
                    **self.plot_params['sf_in_desi']
                )
                ax[4].scatter(
                    self.data.confirmed_sf_and_desi[col_x],
                    self.data.confirmed_sf_and_desi['FEH'],
                    **self.plot_params['sf_in_desi']
                )
            
            if hasattr(self.data, 'cut_confirmed_sf_and_desi') and show_cut:
                ax[0].scatter(
                    self.data.cut_confirmed_sf_and_desi[col_x],
                    self.data.cut_confirmed_sf_and_desi[col_y0],
                    **self.plot_params['sf_in_desi_notsel']
                )
                ax[1].scatter(
                    self.data.cut_confirmed_sf_and_desi[col_x],
                    self.data.cut_confirmed_sf_and_desi['VGSR'],
                    **self.plot_params['sf_in_desi_notsel']
                )
                ax[2].scatter(
                    self.data.cut_confirmed_sf_and_desi[col_x],
                    self.data.cut_confirmed_sf_and_desi['PMRA'],
                    **self.plot_params['sf_in_desi_notsel']
                )
                ax[3].scatter(
                    self.data.cut_confirmed_sf_and_desi[col_x],
                    self.data.cut_confirmed_sf_and_desi['PMDEC'],
                    **self.plot_params['sf_in_desi_notsel']
                )
                ax[4].scatter(
                    self.data.cut_confirmed_sf_and_desi[col_x],
                    self.data.cut_confirmed_sf_and_desi['FEH'],
                    **self.plot_params['sf_in_desi_notsel']
                )
                
            if stream_frame and galstream and hasattr(self.data, 'SoI_galstream') and self.data.SoI_galstream is not None:
                ax[0].plot(
                    self.data.SoI_galstream.gal_phi1,
                    self.data.SoI_galstream.gal_phi2,
                    **self.plot_params['galstream_track']
                )
            elif not stream_frame and galstream and hasattr(self.data, 'SoI_galstream') and self.data.SoI_galstream is not None:
                ax[0].plot(
                    self.data.SoI_galstream.track.ra,
                    self.data.SoI_galstream.track.dec,
                    **self.plot_params['galstream_track']
                )
                
        if show_sf_only:
            ax[0].scatter(
                self.data.confirmed_sf_not_desi[col_x_],
                self.data.confirmed_sf_not_desi[col_y0_],
                **self.plot_params['sf_not_desi']
            )
            ax[1].scatter(
                self.data.confirmed_sf_not_desi[col_x_],
                self.data.confirmed_sf_not_desi['VGSR'],
                **self.plot_params['sf_not_desi']
            )
            ax[2].scatter(
                self.data.confirmed_sf_not_desi[col_x_],
                self.data.confirmed_sf_not_desi['pmRA'],
                **self.plot_params['sf_not_desi']
            )
            ax[3].scatter(
                self.data.confirmed_sf_not_desi[col_x_],
                self.data.confirmed_sf_not_desi['pmDE'],
                **self.plot_params['sf_not_desi']
            )
            # Note: FEH not available in sf_not_desi, skip this subplot
            
        if background:
            ax[0].scatter(
                self.data.desi_data[col_x],
                self.data.desi_data[col_y0],
                **self.plot_params['background']
            )
            ax[1].scatter(
                self.data.desi_data[col_x],
                self.data.desi_data['VGSR'],
                **self.plot_params['background']
            )
            ax[2].scatter(
                self.data.desi_data[col_x],
                self.data.desi_data['PMRA'],
                **self.plot_params['background']
            )
            ax[3].scatter(
                self.data.desi_data[col_x],
                self.data.desi_data['PMDEC'],
                **self.plot_params['background']
            )
            ax[4].scatter(
                self.data.desi_data[col_x],
                self.data.desi_data['FEH'],
                **self.plot_params['background']
            )

        # Set y-axis limits based on stream data if available
        if showStream and hasattr(self.data, 'confirmed_sf_and_desi') and len(self.data.confirmed_sf_and_desi) > 0:
            # VGSR limits
            vgsr_data = [self.data.confirmed_sf_and_desi['VGSR']]
            if hasattr(self.data, 'cut_confirmed_sf_and_desi') and len(self.data.cut_confirmed_sf_and_desi) > 0 and show_cut:
                vgsr_data.append(self.data.cut_confirmed_sf_and_desi['VGSR'])
            vgsr_combined = np.concatenate(vgsr_data)
            ax[1].set_ylim(np.nanmin(vgsr_combined) - 50, np.nanmax(vgsr_combined) + 50)
            
            # Proper motion limits  
            pmra_data = [self.data.confirmed_sf_and_desi['PMRA']]
            pmdec_data = [self.data.confirmed_sf_and_desi['PMDEC']]
            if hasattr(self.data, 'cut_confirmed_sf_and_desi') and len(self.data.cut_confirmed_sf_and_desi) > 0 and show_cut:
                pmra_data.append(self.data.cut_confirmed_sf_and_desi['PMRA'])
                pmdec_data.append(self.data.cut_confirmed_sf_and_desi['PMDEC'])
            pmra_combined = np.concatenate(pmra_data)
            pmdec_combined = np.concatenate(pmdec_data)
            ax[2].set_ylim(np.nanmin(pmra_combined) - 5, np.nanmax(pmra_combined) + 5)
            ax[3].set_ylim(np.nanmin(pmdec_combined) - 5, np.nanmax(pmdec_combined) + 5)

        # Set metallicity limits
        ax[4].set_ylim(-4, -0.5)
        
        # Plot splines if requested and available
        if (show_initial_splines or show_optimized_splines) and stream_frame and self.mcmeta is not None:
            # Create phi1 range for spline plotting
            phi1_min = ax[1].get_xlim()[0]
            phi1_max = ax[1].get_xlim()[1]
            phi1_spline_plot = np.linspace(phi1_min, phi1_max, 100)
            
            # Plot initial guess splines in black
            if show_initial_splines:
                if hasattr(self.mcmeta, 'phi1_spline_points'):
                    try:
                        # VGSR spline
                        vgsr_initial = stream_funcs.apply_spline(
                            phi1_spline_plot, self.mcmeta.phi1_spline_points, 
                            self.mcmeta.initial_params['vgsr_spline_points'], k=2
                        )
                        ax[1].plot(phi1_spline_plot, vgsr_initial, 'k-', linewidth=2, 
                                  label='Initial Spline', alpha=0.8)
                        
                        # PMRA spline
                        pmra_initial = stream_funcs.apply_spline(
                            phi1_spline_plot, self.mcmeta.phi1_spline_points, 
                            self.mcmeta.initial_params['pmra_spline_points'], k=2
                        )
                        ax[2].plot(phi1_spline_plot, pmra_initial, 'k-', linewidth=2, 
                                  label='Initial Spline', alpha=0.8)
                        
                        # PMDEC spline
                        pmdec_initial = stream_funcs.apply_spline(
                            phi1_spline_plot, self.mcmeta.phi1_spline_points, 
                            self.mcmeta.initial_params['pmdec_spline_points'], k=2
                        )
                        ax[3].plot(phi1_spline_plot, pmdec_initial, 'k-', linewidth=2, 
                                  label='Initial Spline', alpha=0.8)
                        
                        # FEH constant line
                        feh_initial = np.full_like(phi1_spline_plot, self.mcmeta.initial_params['feh1'])
                        ax[4].plot(phi1_spline_plot, feh_initial, 'k-', linewidth=2, 
                                  label='Initial [Fe/H]', alpha=0.8)
                    except Exception as e:
                        print(f"Warning: Could not plot initial splines: {e}")
            
            # Plot optimized splines in red
            if show_optimized_splines and hasattr(self.mcmeta, 'optimized_params'):
                if hasattr(self.mcmeta, 'phi1_spline_points'):
                    try:
                        # VGSR spline
                        vgsr_optimized = stream_funcs.apply_spline(
                            phi1_spline_plot, self.mcmeta.phi1_spline_points, 
                            self.mcmeta.optimized_params['vgsr_spline_points'], k=2
                        )
                        ax[1].plot(phi1_spline_plot, vgsr_optimized, 'r-', linewidth=2, 
                                  label='Optimized Spline', alpha=0.8)
                        
                        # PMRA spline
                        pmra_optimized = stream_funcs.apply_spline(
                            phi1_spline_plot, self.mcmeta.phi1_spline_points, 
                            self.mcmeta.optimized_params['pmra_spline_points'], k=2
                        )
                        ax[2].plot(phi1_spline_plot, pmra_optimized, 'r-', linewidth=2, 
                                  label='Optimized Spline', alpha=0.8)
                        
                        # PMDEC spline
                        pmdec_optimized = stream_funcs.apply_spline(
                            phi1_spline_plot, self.mcmeta.phi1_spline_points, 
                            self.mcmeta.optimized_params['pmdec_spline_points'], k=2
                        )
                        ax[3].plot(phi1_spline_plot, pmdec_optimized, 'r-', linewidth=2, 
                                  label='Optimized Spline', alpha=0.8)
                        
                        # FEH constant line
                        feh_optimized = np.full_like(phi1_spline_plot, self.mcmeta.optimized_params['feh1'])
                        ax[4].plot(phi1_spline_plot, feh_optimized, 'r-', linewidth=2, 
                                  label='Optimized [Fe/H]', alpha=0.8)
                    except Exception as e:
                        print(f"Warning: Could not plot optimized splines: {e}")
            
            # Plot MCMC splines in blue (requires external meds dictionary with MCMC results)
            if show_mcmc_splines and hasattr(self.mcmeta, 'phi1_spline_points'):
                try:
                    # Try to access meds from the global namespace or pass it as parameter
                    # This is a bit of a hack - in a proper implementation you'd pass meds as a parameter
                    import inspect
                    frame = inspect.currentframe()
                    try:
                        # Look for meds in the calling frame's globals
                        caller_frame = frame.f_back
                        while caller_frame:
                            if 'meds' in caller_frame.f_globals:
                                meds = caller_frame.f_globals['meds']
                                break
                            elif 'meds' in caller_frame.f_locals:
                                meds = caller_frame.f_locals['meds']
                                break
                            caller_frame = caller_frame.f_back
                        else:
                            raise NameError("meds not found in calling context")
                            
                        # Extract spline points from meds dictionary
                        no_of_spline_points = len(self.mcmeta.phi1_spline_points)
                        
                        # Build the spline arrays from the flattened meds
                        vgsr_mcmc_points = []
                        pmra_mcmc_points = []  
                        pmdec_mcmc_points = []
                        
                        # Extract vgsr spline points
                        for i in range(1, no_of_spline_points + 1):
                            vgsr_mcmc_points.append(meds[f'vgsr{i}'])
                            
                        # Extract pmra spline points  
                        for i in range(1, no_of_spline_points + 1):
                            pmra_mcmc_points.append(meds[f'pmra{i}'])
                            
                        # Extract pmdec spline points
                        for i in range(1, no_of_spline_points + 1):
                            pmdec_mcmc_points.append(meds[f'pmdec{i}'])
                        
                        # Convert to arrays
                        vgsr_mcmc_points = np.array(vgsr_mcmc_points)
                        pmra_mcmc_points = np.array(pmra_mcmc_points)  
                        pmdec_mcmc_points = np.array(pmdec_mcmc_points)
                        
                        # VGSR spline
                        vgsr_mcmc = stream_funcs.apply_spline(
                            phi1_spline_plot, self.mcmeta.phi1_spline_points, 
                            vgsr_mcmc_points, k=2
                        )
                        ax[1].plot(phi1_spline_plot, vgsr_mcmc, 'b-', linewidth=2, 
                                  label='MCMC Spline', alpha=0.8)
                        
                        # PMRA spline
                        pmra_mcmc = stream_funcs.apply_spline(
                            phi1_spline_plot, self.mcmeta.phi1_spline_points, 
                            pmra_mcmc_points, k=2
                        )
                        ax[2].plot(phi1_spline_plot, pmra_mcmc, 'b-', linewidth=2, 
                                  label='MCMC Spline', alpha=0.8)
                        
                        # PMDEC spline
                        pmdec_mcmc = stream_funcs.apply_spline(
                            phi1_spline_plot, self.mcmeta.phi1_spline_points, 
                            pmdec_mcmc_points, k=2
                        )
                        ax[3].plot(phi1_spline_plot, pmdec_mcmc, 'b-', linewidth=2, 
                                  label='MCMC Spline', alpha=0.8)
                        
                        # FEH constant line
                        feh_mcmc = np.full_like(phi1_spline_plot, meds['feh1'])
                        ax[4].plot(phi1_spline_plot, feh_mcmc, 'b-', linewidth=2, 
                                  label='MCMC [Fe/H]', alpha=0.8)
                                  
                    finally:
                        del frame
                        
                except Exception as e:
                    print(f"Warning: Could not plot MCMC splines: {e}")
                    print("Make sure 'meds' dictionary with MCMC results is available in the calling scope")
        
        # Labels and formatting
        if show_initial_splines or show_optimized_splines or show_mcmc_splines:
            # Add legends to kinematic plots if splines are shown
            for i in [1]:  # Show legends on all kinematic plots
                if ax[i].get_lines():  # Only add legend if there are lines to show
                    ax[i].legend(loc='best', fontsize='small')
        
        ax[0].legend(loc='upper left', ncol=4)
        ax[0].set_ylabel(label_y0)
        ax[1].set_ylabel(r'V$_{GSR}$ (km/s)')
        ax[2].set_ylabel(r'$\mu_{\alpha}$ [mas/yr]')
        ax[3].set_ylabel(r'$\mu_{\delta}$ [mas/yr]')
        ax[4].set_ylabel(r'[Fe/H]')
        ax[-1].set_xlabel(label_x)
        
        for a in ax:
            stream_funcs.plot_form(a)
            
        if save:
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}sixD_plot_{self.stream.streamName}.png", dpi=300, bbox_inches='tight')
            
        return fig, ax
    
    def gaussian_mixture_plot(self, showStream=True, show_sf_only=False, background=True, save=False):
        """
        Plots Gaussian mixture model distributions for stream vs background in 4 dimensions:
        VGSR, FEH, PMRA, PMDEC
        
        Uses truncated Gaussians based on the selection cuts applied to the data.
        Requires MCMeta object to be initialized with initial parameters.
        """
        if self.mcmeta is None:
            raise ValueError("MCMeta object required for gaussian_mixture_plot. Initialize StreamPlotter with MCMeta object.")
            
        colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        from scipy.stats import truncnorm
        
        fig, axes = plt.subplots(2, 2, figsize=(9, 9))
        
        # Get data arrays
        desi_data = self.data.desi_data
        sf_data = self.data.confirmed_sf_and_desi if hasattr(self.data, 'confirmed_sf_and_desi') else pd.DataFrame()
        
        # Define plotting parameters
        alpha_stream = 0.7
        alpha_bg = 0.5
        bins = 50
        
        # Estimate mixture weights
        n_stream = len(sf_data) if len(sf_data) > 0 else 1
        n_total = len(desi_data)
        stream_weight = n_stream / n_total
        bg_weight = 1 - stream_weight
        
        # VGSR plot (top left)
        ax = axes[0, 0]
        if background:
            ax.hist(desi_data['VGSR'], density=True, color='lightgrey', bins=bins, alpha=0.7, label='DESI Data')
        
        if showStream and len(sf_data) > 0:
            ax.hist(sf_data['VGSR'], density=True, color='lightblue', bins=bins, alpha=0.8, label='SF Stars')
            
        # Plot truncated Gaussian components
        vgsr_range = np.linspace(self.mcmeta.truncation_params['vgsr_min'] - 50, 
                                self.mcmeta.truncation_params['vgsr_max'] + 50, 200)
        
        # Stream component (using mean of spline points as approximation)
        stream_vgsr_mean = np.mean(self.mcmeta.initial_params['vgsr_spline_points'])
        stream_vgsr_std = 10**self.mcmeta.initial_params['lsigvgsr']
        
        # Background component
        bg_vgsr_mean = self.mcmeta.initial_params['bv']
        bg_vgsr_std = 10**self.mcmeta.initial_params['lsigbv']
        
        # Stream truncated normal
        stream_a = (self.mcmeta.truncation_params['vgsr_min'] - stream_vgsr_mean) / stream_vgsr_std
        stream_b = (self.mcmeta.truncation_params['vgsr_max'] - stream_vgsr_mean) / stream_vgsr_std
        stream_vgsr_pdf = truncnorm.pdf(vgsr_range, stream_a, stream_b, loc=stream_vgsr_mean, scale=stream_vgsr_std)
        
        # Background truncated normal
        bg_a = (self.mcmeta.truncation_params['vgsr_min'] - bg_vgsr_mean) / bg_vgsr_std
        bg_b = (self.mcmeta.truncation_params['vgsr_max'] - bg_vgsr_mean) / bg_vgsr_std
        bg_vgsr_pdf = truncnorm.pdf(vgsr_range, bg_a, bg_b, loc=bg_vgsr_mean, scale=bg_vgsr_std)
        
        ax.plot(vgsr_range, stream_weight * stream_vgsr_pdf, ':', color=colors[0], label='Stream Component', lw=3)
        ax.plot(vgsr_range, bg_weight * bg_vgsr_pdf, ':', color=colors[1], label='Background Component', lw=3)
        ax.plot(vgsr_range, stream_weight * stream_vgsr_pdf + bg_weight * bg_vgsr_pdf, 'k-', label='Total Model', lw=3)
        
        ax.set_xlabel(r'V$_{GSR}$ (km/s)', fontsize=12)
        ax.set_xlim(self.mcmeta.truncation_params['vgsr_min'] - 50, self.mcmeta.truncation_params['vgsr_max'] + 50)
        ax.legend(fontsize='large')
        ax.tick_params(axis='both', labelsize=14)
        stream_funcs.plot_form(ax)
        
        # FEH plot (top right)
        ax = axes[0, 1]
        if background:
            ax.hist(desi_data['FEH'], density=True, color='lightgrey', bins=bins, alpha=0.7)
            
        if showStream and len(sf_data) > 0:
            ax.hist(sf_data['FEH'], density=True, color='lightblue', bins=bins, alpha=0.8)
            
        # Plot truncated Gaussian components
        feh_range = np.linspace(self.mcmeta.truncation_params['feh_min'] - 0.5, 
                               self.mcmeta.truncation_params['feh_max'] + 0.5, 200)
        
        # Stream component
        stream_feh_mean = self.mcmeta.initial_params['feh1']
        stream_feh_std = 10**self.mcmeta.initial_params['lsigfeh']
        
        # Background component
        bg_feh_mean = self.mcmeta.initial_params['bfeh']
        bg_feh_std = 10**self.mcmeta.initial_params['lsigbfeh']
        
        # Stream truncated normal
        stream_a = (self.mcmeta.truncation_params['feh_min'] - stream_feh_mean) / stream_feh_std
        stream_b = (self.mcmeta.truncation_params['feh_max'] - stream_feh_mean) / stream_feh_std
        stream_feh_pdf = truncnorm.pdf(feh_range, stream_a, stream_b, loc=stream_feh_mean, scale=stream_feh_std)
        
        # Background truncated normal
        bg_a = (self.mcmeta.truncation_params['feh_min'] - bg_feh_mean) / bg_feh_std
        bg_b = (self.mcmeta.truncation_params['feh_max'] - bg_feh_mean) / bg_feh_std
        bg_feh_pdf = truncnorm.pdf(feh_range, bg_a, bg_b, loc=bg_feh_mean, scale=bg_feh_std)
        
        ax.plot(feh_range, stream_weight * stream_feh_pdf, ':', color=colors[0], lw=3)
        ax.plot(feh_range, bg_weight * bg_feh_pdf, ':', color=colors[1], lw=3)
        ax.plot(feh_range, stream_weight * stream_feh_pdf + bg_weight * bg_feh_pdf, 'k-', lw=3)
        
        ax.set_xlabel('[Fe/H]', fontsize=12)
        ax.set_xlim(self.mcmeta.truncation_params['feh_min'] - 0.5, self.mcmeta.truncation_params['feh_max'] + 0.5)
        ax.tick_params(axis='both', labelsize=14)
        stream_funcs.plot_form(ax)
        
        # PMRA plot (bottom left)
        ax = axes[1, 0]
        if background:
            ax.hist(desi_data['PMRA'], density=True, color='lightgrey', bins=bins, alpha=0.7)
            
        if showStream and len(sf_data) > 0:
            ax.hist(sf_data['PMRA'], density=True, color='lightblue', bins=bins, alpha=0.8)
            
        # Plot truncated Gaussian components
        pmra_range = np.linspace(self.mcmeta.truncation_params['pmra_min'] - 15, 
                                self.mcmeta.truncation_params['pmra_max'] + 15, 200)
        
        # Stream component (using mean of spline points as approximation)
        stream_pmra_mean = np.mean(self.mcmeta.initial_params['pmra_spline_points'])
        stream_pmra_std = 10**self.mcmeta.initial_params['lsigpmra']
        
        # Background component
        bg_pmra_mean = self.mcmeta.initial_params['bpmra']
        bg_pmra_std = 10**self.mcmeta.initial_params['lsigbpmra']
        
        # Stream truncated normal
        stream_a = (self.mcmeta.truncation_params['pmra_min'] - stream_pmra_mean) / stream_pmra_std
        stream_b = (self.mcmeta.truncation_params['pmra_max'] - stream_pmra_mean) / stream_pmra_std
        stream_pmra_pdf = truncnorm.pdf(pmra_range, stream_a, stream_b, loc=stream_pmra_mean, scale=stream_pmra_std)
        
        # Background truncated normal
        bg_a = (self.mcmeta.truncation_params['pmra_min'] - bg_pmra_mean) / bg_pmra_std
        bg_b = (self.mcmeta.truncation_params['pmra_max'] - bg_pmra_mean) / bg_pmra_std
        bg_pmra_pdf = truncnorm.pdf(pmra_range, bg_a, bg_b, loc=bg_pmra_mean, scale=bg_pmra_std)
        
        ax.plot(pmra_range, stream_weight * stream_pmra_pdf, ':', color=colors[0], lw=3)
        ax.plot(pmra_range, bg_weight * bg_pmra_pdf, ':', color=colors[1], lw=3)
        ax.plot(pmra_range, stream_weight * stream_pmra_pdf + bg_weight * bg_pmra_pdf, 'k-', lw=3)
        
 
        ax.set_xlabel(r'$\mu_{RA}$ (mas/yr)', fontsize=12)
        ax.set_xlim(self.mcmeta.truncation_params['pmra_min'] - 15, self.mcmeta.truncation_params['pmra_max'] + 15)
        ax.tick_params(axis='both', labelsize=14)
        stream_funcs.plot_form(ax)
        
        # PMDEC plot (bottom right)
        ax = axes[1, 1]
        if background:
            ax.hist(desi_data['PMDEC'], density=True, color='lightgrey', bins=bins, alpha=0.7)
            
        if showStream and len(sf_data) > 0:
            ax.hist(sf_data['PMDEC'], density=True, color='lightblue', bins=bins, alpha=0.8)
            
        # Plot truncated Gaussian components
        pmdec_range = np.linspace(self.mcmeta.truncation_params['pmdec_min'] - 15, 
                                 self.mcmeta.truncation_params['pmdec_max'] + 15, 200)
        
        # Stream component (using mean of spline points as approximation)
        stream_pmdec_mean = np.mean(self.mcmeta.initial_params['pmdec_spline_points'])
        stream_pmdec_std = 10**self.mcmeta.initial_params['lsigpmdec']
        
        # Background component
        bg_pmdec_mean = self.mcmeta.initial_params['bpmdec']
        bg_pmdec_std = 10**self.mcmeta.initial_params['lsigbpmdec']
        
        # Stream truncated normal
        stream_a = (self.mcmeta.truncation_params['pmdec_min'] - stream_pmdec_mean) / stream_pmdec_std
        stream_b = (self.mcmeta.truncation_params['pmdec_max'] - stream_pmdec_mean) / stream_pmdec_std
        stream_pmdec_pdf = truncnorm.pdf(pmdec_range, stream_a, stream_b, loc=stream_pmdec_mean, scale=stream_pmdec_std)
        
        # Background truncated normal
        bg_a = (self.mcmeta.truncation_params['pmdec_min'] - bg_pmdec_mean) / bg_pmdec_std
        bg_b = (self.mcmeta.truncation_params['pmdec_max'] - bg_pmdec_mean) / bg_pmdec_std
        bg_pmdec_pdf = truncnorm.pdf(pmdec_range, bg_a, bg_b, loc=bg_pmdec_mean, scale=bg_pmdec_std)
        
        ax.plot(pmdec_range, stream_weight * stream_pmdec_pdf, ':', color=colors[0], lw=3)
        ax.plot(pmdec_range, bg_weight * bg_pmdec_pdf, ':', color=colors[1], lw=3)
        ax.plot(pmdec_range, stream_weight * stream_pmdec_pdf + bg_weight * bg_pmdec_pdf, 'k-', lw=3)
        
        ax.set_xlabel(r'$\mu_{DEC}$ (mas/yr)', fontsize=12)
        ax.set_xlim(self.mcmeta.truncation_params['pmdec_min'] - 15, self.mcmeta.truncation_params['pmdec_max'] + 15)
        ax.tick_params(axis='both', labelsize=14)
        stream_funcs.plot_form(ax)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.save_dir}gaussian_mixture_{self.stream.streamName}.png", dpi=300, bbox_inches='tight')
            
        return fig, axes


class MCMeta:
    """
    For creating and plotting a spline track of the stream.
    """
    def __init__(self, no_of_spline_points, stream_object, sf_data, truncation_params=None):
        self.stream = stream_object
        self.no_of_spline_points = no_of_spline_points
        self.sf_data = sf_data
        if self.no_of_spline_points == 1:
            self.spline_k = 1
        elif self.no_of_spline_points > 3:
            self.spline_k = 3
        else:
            self.spline_k = self.no_of_spline_points - 1

        self.phi1_spline_points = np.linspace(self.stream.data.SoI_streamfinder['phi1'].min()-5, self.stream.data.SoI_streamfinder['phi1'].max()+5, self.no_of_spline_points)

        # Store truncation parameters for plotting
        if truncation_params is None:
            # Default truncation values based on data range
            data = self.stream.data.desi_data
            self.truncation_params = {
                'vgsr_min': data['VGSR'].min(),
                'vgsr_max': data['VGSR'].max(),
                'feh_min': data['FEH'].min(),
                'feh_max': data['FEH'].max(),
                'pmra_min': data['PMRA'].min(),
                'pmra_max': data['PMRA'].max(),
                'pmdec_min': data['PMDEC'].min(),
                'pmdec_max': data['PMDEC'].max()
            }
        else:
            self.truncation_params = truncation_params

        self.param_labels = [
            "vgsr_spline_points", "lsigvgsr",
            "feh1", "lsigfeh",
            "pmra_spline_points", "lsigpmra",
            "pmdec_spline_points", "lsigpmdec",
            "bv", "lsigbv", "bfeh", "lsigbfeh", "bpmra", "lsigbpmra", "bpmdec", "lsigbpmdec"
        ]

        # Initialize the initial_params dictionary
        self.initial_params = {}

        print('Making stream initial guess based on galstream and STREAMFINDER...')
        p = np.polyfit(self.sf_data['phi1'].values, self.sf_data['VGSR'].values, 2)
        self.vgsr_fit = np.poly1d(p)
        self.initial_params['lsigvgsr'] = np.log10(self.sf_data['VGSR'].values.std())
        self.initial_params['vgsr_spline_points'] = self.vgsr_fit(self.phi1_spline_points)
        print(f"Stream VGSR dispersion from trimmed SF: {10**self.initial_params['lsigvgsr']:.2f} km/s")

        self.initial_params['feh1'] = self.sf_data['FEH'].values.mean()
        self.initial_params['lsigfeh'] = np.log10(self.sf_data['FEH'].values.std())
        print(f'Stream mean metallicity from trimmed SF: {self.initial_params["feh1"]:.2f} +- {10**self.initial_params["lsigfeh"]:.3f} dex')

        p = np.polyfit(self.sf_data['phi1'].values, self.sf_data['PMRA'].values, 2)
        self.pmra_fit = np.poly1d(p)
        self.initial_params['lsigpmra'] = np.log10(self.sf_data['PMRA'].values.std())
        self.initial_params['pmra_spline_points'] = self.pmra_fit(self.phi1_spline_points)
        print(f"Stream PMRA dispersion from trimmed SF: {10**self.initial_params['lsigpmra']:.2f} mas/yr")

        p = np.polyfit(self.sf_data['phi1'].values, self.sf_data['PMDEC'].values, 2)
        self.pmdec_fit = np.poly1d(p)
        self.initial_params['lsigpmdec'] = np.log10(self.sf_data['PMDEC'].values.std())
        self.initial_params['pmdec_spline_points'] = self.pmdec_fit(self.phi1_spline_points)
        print(f"Stream PMDEC dispersion from trimmed SF: {10**self.initial_params['lsigpmdec']:.2f} mas/yr")

        print('Making background initial guess...')
        self.initial_params['bv'] = np.mean(np.array(self.stream.data.desi_data['VGSR']))
        self.initial_params['lsigbv'] = np.log10(np.std(np.array(self.stream.data.desi_data['VGSR'])))
        print(f"Background velocity: {self.initial_params['bv']:.2f} +- {10**self.initial_params['lsigbv']:.2f} km/s")

        self.initial_params['bfeh'] = np.mean(np.array(self.stream.data.desi_data['FEH']))
        self.initial_params['lsigbfeh'] = np.log10(np.std(np.array(self.stream.data.desi_data['FEH'])))
        print(f"Background metallicity: {self.initial_params['bfeh']:.2f} +- {10**self.initial_params['lsigbfeh']:.3f} dex")

        self.initial_params['bpmra'] = np.mean(np.array(self.stream.data.desi_data['PMRA']))
        self.initial_params['lsigbpmra'] = np.log10(np.std(np.array(self.stream.data.desi_data['PMRA'])))
        print(f"Background PMRA: {self.initial_params['bpmra']:.2f} +- {10**self.initial_params['lsigbpmra']:.2f} mas/yr")

        self.initial_params['bpmdec'] = np.mean(np.array(self.stream.data.desi_data['PMDEC']))
        self.initial_params['lsigbpmdec'] = np.log10(np.std(np.array(self.stream.data.desi_data['PMDEC'])))
        print(f"Background PMDEC: {self.initial_params['bpmdec']:.2f} +- {10**self.initial_params['lsigbpmdec']:.2f} mas/yr")

    def create_plotter(self, save_dir='plots/'):
        """
        Create a StreamPlotter object initialized with this MCMeta instance.
        
        Returns:
            StreamPlotter: A plotter object that can access both stream and MCMeta functionality.
        """
        return StreamPlotter(self, save_dir=save_dir)

    def plot_chains(self, sampler, param_labels, save=False, burnin=0, thin=1, figsize=(12, 8)):
        """
        Plot MCMC chains for all parameters to check convergence
        
        Parameters:
        -----------
        sampler : emcee.EnsembleSampler
            The emcee sampler object from MCMC run
        param_labels : list
            List of parameter names for labeling
        save : bool, optional
            Whether to save the plot. Default is False.
        burnin : int, optional
            Number of burn-in samples to discard. Default is 0.
        thin : int, optional
            Thin the chain by this factor. Default is 1.
        figsize : tuple, optional
            Figure size. Default is (12, 8).
            
        Returns:
        --------
        fig, axes : matplotlib figure and axes
            The plot figure and axes
        """
        # Get the chain
        samples = sampler.get_chain(discard=burnin, thin=thin)
        nwalkers, nsteps, ndim = samples.shape
        
        # Create subplots
        ncols = 3
        nrows = int(np.ceil(ndim / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        # Handle different subplot configurations
        if nrows == 1 and ncols == 1:
            axes = [axes]
        elif nrows == 1 or ncols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Plot each parameter
        for i in range(ndim):
            ax = axes[i]
            
            # Plot all walker chains
            for j in range(nwalkers):
                ax.plot(samples[j, :, i], alpha=0.3, color='k', lw=0.5)
            
            # Set labels
            param_name = param_labels[i] if i < len(param_labels) else f'param_{i}'
            ax.set_title(param_name, fontsize=10)
            ax.set_xlabel('Step')
            ax.set_ylabel('Value')
            stream_funcs.plot_form(ax)
        
        # Hide unused subplots
        for i in range(ndim, len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        
        if save:
            save_dir = 'plots/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(f"{save_dir}mcmc_chains_{self.stream.streamName}.png", 
                       dpi=300, bbox_inches='tight')
        
        return fig, axes

    def plot_corner(self, sampler, param_labels, save=False, burnin=0, thin=1, truths=None, 
                   show_titles=True, title_fmt=".3f", quantiles=[0.16, 0.5, 0.84]):
        """
        Create a corner plot of the MCMC posterior distributions
        
        Parameters:
        -----------
        sampler : emcee.EnsembleSampler
            The emcee sampler object from MCMC run
        param_labels : list
            List of parameter names for labeling
        save : bool, optional
            Whether to save the plot. Default is False.
        burnin : int, optional
            Number of burn-in samples to discard. Default is 0.
        thin : int, optional
            Thin the chain by this factor. Default is 1.
        truths : array_like, optional
            True parameter values to overplot. Default is None.
        show_titles : bool, optional
            Whether to show parameter value titles. Default is True.
        title_fmt : str, optional
            Format string for parameter titles. Default is ".3f".
        quantiles : list, optional
            Quantiles to show in histograms. Default is [0.16, 0.5, 0.84].
            
        Returns:
        --------
        fig : matplotlib figure
            The corner plot figure
        """
        # Get the flattened chain
        samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
        
        # Create the corner plot
        fig = corner.corner(
            samples, 
            labels=param_labels,
            truths=truths,
            show_titles=show_titles,
            title_fmt=title_fmt,
            quantiles=quantiles,
            plot_datapoints=False,
            fill_contours=True,
            levels=(0.68, 0.95),
            color='blue',
            hist_kwargs={'density': True}
        )
        
        if save:
            save_dir = 'plots/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(f"{save_dir}corner_plot_{self.stream.streamName}.png", 
                       dpi=300, bbox_inches='tight')
        
        return fig

    def get_parameter_summary(self, sampler, param_labels, burnin=0, thin=1, percentiles=[16, 50, 84]):
        """
        Get summary statistics for MCMC parameters
        
        Parameters:
        -----------
        sampler : emcee.EnsembleSampler
            The emcee sampler object from MCMC run
        param_labels : list
            List of parameter names
        burnin : int, optional
            Number of burn-in samples to discard. Default is 0.
        thin : int, optional
            Thin the chain by this factor. Default is 1.
        percentiles : list, optional
            Percentiles to compute. Default is [16, 50, 84] for median and 1-sigma.
            
        Returns:
        --------
        dict : parameter summary
            Dictionary with parameter names as keys and [lower, median, upper] as values
        """
        # Get the flattened chain
        samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
        
        summary = {}
        for i, label in enumerate(param_labels):
            if i < samples.shape[1]:
                percentile_values = np.percentile(samples[:, i], percentiles)
                summary[label] = {
                    'lower': percentile_values[0],
                    'median': percentile_values[1], 
                    'upper': percentile_values[2],
                    'mean': np.mean(samples[:, i]),
                    'std': np.std(samples[:, i])
                }
            
        return summary

    def print_parameter_summary(self, sampler, param_labels, burnin=0, thin=1):
        """
        Print a formatted summary of MCMC parameter results
        
        Parameters:
        -----------
        sampler : emcee.EnsembleSampler
            The emcee sampler object from MCMC run
        param_labels : list
            List of parameter names
        burnin : int, optional
            Number of burn-in samples to discard. Default is 0.
        thin : int, optional
            Thin the chain by this factor. Default is 1.
        """
        summary = self.get_parameter_summary(sampler, param_labels, burnin=burnin, thin=thin)
        
        print("\nMCMC Parameter Summary:")
        print("=" * 60)
        print(f"{'Parameter':<20} {'Median':<12} {'Lower':<12} {'Upper':<12}")
        print("-" * 60)
        
        for param, stats in summary.items():
            median = stats['median']
            lower_err = stats['median'] - stats['lower']
            upper_err = stats['upper'] - stats['median']
            
            print(f"{param:<20} {median:>11.4f} -{lower_err:>10.4f} +{upper_err:>10.4f}")
            
        print("=" * 60)

    def get_best_fit_parameters(self, sampler, param_labels, burnin=0, thin=1, method='median'):
        """
        Get the best-fit parameters from MCMC chain
        
        Parameters:
        -----------
        sampler : emcee.EnsembleSampler
            The emcee sampler object from MCMC run
        param_labels : list
            List of parameter names
        burnin : int, optional
            Number of burn-in samples to discard. Default is 0.
        thin : int, optional
            Thin the chain by this factor. Default is 1.
        method : str, optional
            Method to determine best-fit: 'median', 'mean', or 'max_likelihood'. Default is 'median'.
            
        Returns:
        --------
        dict : best-fit parameters
            Dictionary with parameter names as keys and best-fit values
        """
        # Get the flattened chain
        samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
        
        if method == 'median':
            best_fit = {label: np.median(samples[:, i]) for i, label in enumerate(param_labels) if i < samples.shape[1]}
        elif method == 'mean':
            best_fit = {label: np.mean(samples[:, i]) for i, label in enumerate(param_labels) if i < samples.shape[1]}
        elif method == 'max_likelihood':
            # Find sample with maximum likelihood
            log_probs = sampler.get_log_prob(discard=burnin, thin=thin, flat=True)
            best_idx = np.argmax(log_probs)
            best_fit = {label: samples[best_idx, i] for i, label in enumerate(param_labels) if i < samples.shape[1]}
        else:
            raise ValueError("method must be 'median', 'mean', or 'max_likelihood'")
            
        return best_fit