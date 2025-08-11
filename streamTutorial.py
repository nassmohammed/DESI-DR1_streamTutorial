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


        # # Apply Metallicity correction to the DR1 Data (See Section 4.2 https://arxiv.org/pdf/2505.14787)
        # print("Adding empirical FEH calibration (can find uncalibrated data in column['FEH_uncalib])")
        # self.desi_data['FEH_uncalib'] = self.desi_data['FEH']
        # self.desi_data['FEH'] = feh_correct.calibrate(self.desi_data['FEH'], self.desi_data['TEFF'], self.desi_data['LOGG'])
        
        # Switch to VGSR instead of VRAD
        self.desi_data['VGSR'] = np.array(
            stream_funcs.vhel_to_vgsr(
                np.array(self.desi_data['TARGET_RA']) * u.deg,
                np.array(self.desi_data['TARGET_DEC']) * u.deg,
                np.array(self.desi_data['VRAD']) * (u.km / u.s),
            ).value
        )

        # Lets load the STREAMFINDER data for Gaia DR3
        if sf_path:
            sf_data = table.Table.read(self.sf_path)
            self.sf_data = sf_data.to_pandas()
            self.sf_data['VGSR'] = np.array(
                stream_funcs.vhel_to_vgsr(
                    np.array(self.sf_data['RAdeg']) * u.deg,
                    np.array(self.sf_data['DEdeg']) * u.deg,
                    np.array(self.sf_data['VHel']) * (u.km / u.s),
                ).value
            )
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
        print(f'Mass Fraction (Z): {mass_fraction}')

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
        print('')
        use_mass_frac = dotter_mass_frac_str[np.argmin(np.abs(dotter_mass_frac - mass_fraction))]


        isochrone_path = dotter_directory + 'iso_a' + str(age) + '_z' + str(use_mass_frac) + '.dat'
        print(f'using {isochrone_path}')
        dotter_mp = np.loadtxt(isochrone_path)
        self.isochrone_path = isochrone_path

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
            },
            # Centralized styles used in sixD_plot
            'sf_errorbar': {
                'ecolor': 'k',
                'elinewidth': 0.8,
                'capsize': 2,
                'alpha': 0.8,
                'zorder': 4
            },
            'member_errorbar': {
                'ecolor': 'black',
                'elinewidth': 0.8,
                'capsize': 2,
                'alpha': 0.8,
                'zorder': 5
            },
            'initial_spline_line': {
                'color': 'k',
                'lw': 2,
                'alpha': 0.8,
                'label': 'Initial Spline'
            },
            'optimized_spline_line': {
                'color': 'r',
                'lw': 2,
                'alpha': 0.8,
                'label': 'Optimized Spline'
            },
            'initial_feh_line': {
                'color': 'k',
                'lw': 2,
                'alpha': 0.8,
                'label': 'Initial [Fe/H]'
            },
            'optimized_feh_line': {
                'color': 'r',
                'lw': 2,
                'alpha': 0.8,
                'label': 'Optimized [Fe/H]'
            },
            'spline_knot_initial': {
                'marker': 'o',
                'color': 'black',
                's': 50,
                'zorder': 10,
                'alpha': 0.8,
                'edgecolors': 'white',
                'linewidth': 1
            },
            'spline_knot_optimized': {
                'marker': 'o',
                'color': 'red',
                's': 50,
                'zorder': 10,
                'alpha': 0.8,
                'edgecolors': 'white',
                'linewidth': 1
            },
            'membership_scatter': {
                'marker': 'o',
                's': 25,
                'edgecolor': 'black',
                'linewidth': 0.5,
                'alpha': 0.8,
                'zorder': 6
            },
            'sf_high_prob_diamond': {
                'marker': 'D',
                's': 40,
                'edgecolor': 'black',
                'linewidth': 1,
                'alpha': 1.0,
                'zorder': 7
            },
            'sf_low_prob_diamond': {
                'marker': 'D',
                's': 40,
                'color': 'black',
                'edgecolor': 'black',
                'linewidth': 1,
                'alpha': 1.0,
                'zorder': 7
            },
            'membership_colorbar': {
                'cmap': 'viridis',
                'label': 'Membership Probability',
                'pad': 0.02,
                'aspect': 50,
                'shrink': 1.0,
                'location': 'right',
                'labelpad': 15
            },
            'limits': {
                'feh_ylim_default': (-4, -0.5),
                'residual_pad_vgsr': 20,
                'residual_pad_pm': 2,
                'residual_pad_feh': 0.2
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
                  show_initial_splines=False, show_optimized_splines=False, show_mcmc_splines=False, show_sf_errors=True, 
                  show_membership_prob=False, stream_prob=None, min_prob=0.5, show_residuals=False, mcmc_object=None):
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
        show_membership_prob : bool, optional
            Whether to plot high membership probability stars with special styling. Default is False.
        stream_prob : array-like, optional
            Array of membership probabilities for DESI stars. Required if show_membership_prob=True.
        min_prob : float, optional
            Minimum membership probability threshold for highlighting stars. Default is 0.5.
        show_residuals : bool, optional
            If True and show_mcmc_splines=True in stream_frame, plot residuals (data - MCMC spline) for
            VGSR/PMRA/PMDEC/FEH so the MCMC spline lies along y=0. Ignored if MCMC results are unavailable.
        """
        if stream_frame:
            col_x = 'phi1'
            col_x_ = 'phi1'
            label_x = r'$\phi_1$'
        else:
            col_x = 'TARGET_RA'
            col_x_ = 'RAdeg'
            label_x = 'RA (deg)'
        if show_membership_prob:
            fig, ax = plt.subplots(5, 1, figsize=(15, 15))
        else:
            fig, ax = plt.subplots(5, 1, figsize=(10, 15))
        
        # Residuals mode prep (only applicable with MCMC splines in stream frame)
        residual_mode = False
        preds = {'desi': {}, 'sf': {}, 'sf_cut': {}, 'sf_only': {}}
        if mcmc_object is not None:
                meds = mcmc_object.meds
                ep = mcmc_object.ep
                em = mcmc_object.em
        if show_residuals and show_mcmc_splines and stream_frame and hasattr(self, 'mcmeta') and self.mcmeta is not None and hasattr(self.mcmeta, 'phi1_spline_points'):
            try:

                npts = len(self.mcmeta.phi1_spline_points)
                vgsr_knots = np.array([meds[f'vgsr{i}'] for i in range(1, npts+1)])
                pmra_knots = np.array([meds[f'pmra{i}'] for i in range(1, npts+1)])
                pmdec_knots = np.array([meds[f'pmdec{i}'] for i in range(1, npts+1)])
                feh_const = meds['feh1']

                def eval_spline(phi1_vals, knots):
                    return stream_funcs.apply_spline(phi1_vals, self.mcmeta.phi1_spline_points, knots, k=2)

                # DESI predictions
                phi1_desi = self.data.desi_data[col_x].values
                preds['desi']['vgsr'] = eval_spline(phi1_desi, vgsr_knots)
                preds['desi']['pmra'] = eval_spline(phi1_desi, pmra_knots)
                preds['desi']['pmdec'] = eval_spline(phi1_desi, pmdec_knots)
                preds['desi']['feh'] = np.full_like(phi1_desi, feh_const, dtype=float)

                # SF in DESI predictions
                if hasattr(self.data, 'confirmed_sf_and_desi'):
                    phi1_sf = self.data.confirmed_sf_and_desi[col_x].values
                    preds['sf']['vgsr'] = eval_spline(phi1_sf, vgsr_knots)
                    preds['sf']['pmra'] = eval_spline(phi1_sf, pmra_knots)
                    preds['sf']['pmdec'] = eval_spline(phi1_sf, pmdec_knots)
                    preds['sf']['feh'] = np.full_like(phi1_sf, feh_const, dtype=float)

                # Cut SF predictions
                if hasattr(self.data, 'cut_confirmed_sf_and_desi'):
                    phi1_sf_cut = self.data.cut_confirmed_sf_and_desi[col_x].values
                    preds['sf_cut']['vgsr'] = eval_spline(phi1_sf_cut, vgsr_knots)
                    preds['sf_cut']['pmra'] = eval_spline(phi1_sf_cut, pmra_knots)
                    preds['sf_cut']['pmdec'] = eval_spline(phi1_sf_cut, pmdec_knots)
                    preds['sf_cut']['feh'] = np.full_like(phi1_sf_cut, feh_const, dtype=float)

                # SF only (not in DESI)
                if hasattr(self.data, 'confirmed_sf_not_desi'):
                    phi1_sfo = self.data.confirmed_sf_not_desi[col_x_].values
                    preds['sf_only']['vgsr'] = eval_spline(phi1_sfo, vgsr_knots)
                    preds['sf_only']['pmRA'] = eval_spline(phi1_sfo, pmra_knots)
                    preds['sf_only']['pmDE'] = eval_spline(phi1_sfo, pmdec_knots)

                residual_mode = True
            except Exception as e:
                print(f"Warning: Residuals mode disabled (MCMC meds not available): {e}")
        
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
                # Draw error bars first (behind), then overlay member points
                err_params = {'fmt': 'none'}
                err_params.update(self.plot_params.get('sf_errorbar', {}))
                # Only kinematics/abundance panels have measurement errors; no errorbar for phi2
                ax[1].errorbar(
                    self.data.confirmed_sf_and_desi[col_x],
                    (self.data.confirmed_sf_and_desi['VGSR'] - preds['sf'].get('vgsr', 0)) if residual_mode else self.data.confirmed_sf_and_desi['VGSR'],
                    yerr=self.data.confirmed_sf_and_desi['VRAD_ERR'], **err_params
                )
                ax[2].errorbar(
                    self.data.confirmed_sf_and_desi[col_x],
                    (self.data.confirmed_sf_and_desi['PMRA'] - preds['sf'].get('pmra', 0)) if residual_mode else self.data.confirmed_sf_and_desi['PMRA'],
                    yerr=self.data.confirmed_sf_and_desi['PMRA_ERROR'], **err_params
                )
                ax[3].errorbar(
                    self.data.confirmed_sf_and_desi[col_x],
                    (self.data.confirmed_sf_and_desi['PMDEC'] - preds['sf'].get('pmdec', 0)) if residual_mode else self.data.confirmed_sf_and_desi['PMDEC'],
                    yerr=self.data.confirmed_sf_and_desi['PMDEC_ERROR'], **err_params
                )
                ax[4].errorbar(
                    self.data.confirmed_sf_and_desi[col_x],
                    (self.data.confirmed_sf_and_desi['FEH'] - preds['sf'].get('feh', 0)) if residual_mode else self.data.confirmed_sf_and_desi['FEH'],
                    yerr=self.data.confirmed_sf_and_desi['FEH_ERR'], **err_params
                )
                # Now overlay the member points
                ax[0].scatter(
                    self.data.confirmed_sf_and_desi[col_x],
                    self.data.confirmed_sf_and_desi[col_y0],
                    **self.plot_params['sf_in_desi']
                )
                ax[1].scatter(
                    self.data.confirmed_sf_and_desi[col_x],
                    (self.data.confirmed_sf_and_desi['VGSR'] - preds['sf'].get('vgsr', 0)) if residual_mode else self.data.confirmed_sf_and_desi['VGSR'],
                    **self.plot_params['sf_in_desi']
                )
                ax[2].scatter(
                    self.data.confirmed_sf_and_desi[col_x],
                    (self.data.confirmed_sf_and_desi['PMRA'] - preds['sf'].get('pmra', 0)) if residual_mode else self.data.confirmed_sf_and_desi['PMRA'],
                    **self.plot_params['sf_in_desi']
                )
                ax[3].scatter(
                    self.data.confirmed_sf_and_desi[col_x],
                    (self.data.confirmed_sf_and_desi['PMDEC'] - preds['sf'].get('pmdec', 0)) if residual_mode else self.data.confirmed_sf_and_desi['PMDEC'],
                    **self.plot_params['sf_in_desi']
                )
                ax[4].scatter(
                    self.data.confirmed_sf_and_desi[col_x],
                    (self.data.confirmed_sf_and_desi['FEH'] - preds['sf'].get('feh', 0)) if residual_mode else self.data.confirmed_sf_and_desi['FEH'],
                    **self.plot_params['sf_in_desi']
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
                    (self.data.confirmed_sf_and_desi['VGSR'] - preds['sf'].get('vgsr', 0)) if residual_mode else self.data.confirmed_sf_and_desi['VGSR'],
                    **self.plot_params['sf_in_desi']
                )
                ax[2].scatter(
                    self.data.confirmed_sf_and_desi[col_x],
                    (self.data.confirmed_sf_and_desi['PMRA'] - preds['sf'].get('pmra', 0)) if residual_mode else self.data.confirmed_sf_and_desi['PMRA'],
                    **self.plot_params['sf_in_desi']
                )
                ax[3].scatter(
                    self.data.confirmed_sf_and_desi[col_x],
                    (self.data.confirmed_sf_and_desi['PMDEC'] - preds['sf'].get('pmdec', 0)) if residual_mode else self.data.confirmed_sf_and_desi['PMDEC'],
                    **self.plot_params['sf_in_desi']
                )
                ax[4].scatter(
                    self.data.confirmed_sf_and_desi[col_x],
                    (self.data.confirmed_sf_and_desi['FEH'] - preds['sf'].get('feh', 0)) if residual_mode else self.data.confirmed_sf_and_desi['FEH'],
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
                    (self.data.cut_confirmed_sf_and_desi['VGSR'] - preds['sf_cut'].get('vgsr', 0)) if residual_mode else self.data.cut_confirmed_sf_and_desi['VGSR'],
                    **self.plot_params['sf_in_desi_notsel']
                )
                ax[2].scatter(
                    self.data.cut_confirmed_sf_and_desi[col_x],
                    (self.data.cut_confirmed_sf_and_desi['PMRA'] - preds['sf_cut'].get('pmra', 0)) if residual_mode else self.data.cut_confirmed_sf_and_desi['PMRA'],
                    **self.plot_params['sf_in_desi_notsel']
                )
                ax[3].scatter(
                    self.data.cut_confirmed_sf_and_desi[col_x],
                    (self.data.cut_confirmed_sf_and_desi['PMDEC'] - preds['sf_cut'].get('pmdec', 0)) if residual_mode else self.data.cut_confirmed_sf_and_desi['PMDEC'],
                    **self.plot_params['sf_in_desi_notsel']
                )
                ax[4].scatter(
                    self.data.cut_confirmed_sf_and_desi[col_x],
                    (self.data.cut_confirmed_sf_and_desi['FEH'] - preds['sf_cut'].get('feh', 0)) if residual_mode else self.data.cut_confirmed_sf_and_desi['FEH'],
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
                (self.data.confirmed_sf_not_desi['VGSR'] - preds['sf_only'].get('vgsr', 0)) if residual_mode else self.data.confirmed_sf_not_desi['VGSR'],
                **self.plot_params['sf_not_desi']
            )
            ax[2].scatter(
                self.data.confirmed_sf_not_desi[col_x_],
                (self.data.confirmed_sf_not_desi['pmRA'] - preds['sf_only'].get('pmRA', 0)) if residual_mode else self.data.confirmed_sf_not_desi['pmRA'],
                **self.plot_params['sf_not_desi']
            )
            ax[3].scatter(
                self.data.confirmed_sf_not_desi[col_x_],
                (self.data.confirmed_sf_not_desi['pmDE'] - preds['sf_only'].get('pmDE', 0)) if residual_mode else self.data.confirmed_sf_not_desi['pmDE'],
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
                (self.data.desi_data['VGSR'] - preds['desi'].get('vgsr', 0)) if residual_mode else self.data.desi_data['VGSR'],
                **self.plot_params['background']
            )
            ax[2].scatter(
                self.data.desi_data[col_x],
                (self.data.desi_data['PMRA'] - preds['desi'].get('pmra', 0)) if residual_mode else self.data.desi_data['PMRA'],
                **self.plot_params['background']
            )
            ax[3].scatter(
                self.data.desi_data[col_x],
                (self.data.desi_data['PMDEC'] - preds['desi'].get('pmdec', 0)) if residual_mode else self.data.desi_data['PMDEC'],
                **self.plot_params['background']
            )
            ax[4].scatter(
                self.data.desi_data[col_x],
                (self.data.desi_data['FEH'] - preds['desi'].get('feh', 0)) if residual_mode else self.data.desi_data['FEH'],
                **self.plot_params['background']
            )
        
        # Plot membership probability stars if requested
        if show_membership_prob and stream_prob is not None:
            import matplotlib.cm as cm
            import matplotlib.colors as colors
            
            # Validate stream_prob length matches DESI data
            if len(stream_prob) != len(self.data.desi_data):
                raise ValueError(f"stream_prob length ({len(stream_prob)}) must match DESI data length ({len(self.data.desi_data)})")
            
            # Get high probability stars
            high_prob_mask = stream_prob >= min_prob
            high_prob_indices = np.where(high_prob_mask)[0]
            
            if len(high_prob_indices) > 0:
                # Create colormap for membership probabilities from settings
                norm = colors.Normalize(vmin=min_prob, vmax=1.0)
                cmap_name = self.plot_params.get('membership_colorbar', {}).get('cmap', 'viridis')
                cmap = cm.get_cmap(cmap_name)
                
                # Plot high probability DESI stars as circles with viridis colormap
                base_member = self.plot_params.get('membership_scatter', {})
                scatter_params = {**base_member,
                                  'c': stream_prob[high_prob_indices],
                                  'cmap': cmap,
                                  'norm': norm,
                                  'label': f'High Prob Stars (≥{min_prob:.1f})'}
                
                # Prepare y values (observed or residual) for errorbars and scatter
                x_hp = self.data.desi_data[col_x].iloc[high_prob_indices]
                y1_hp = self.data.desi_data['VGSR'].iloc[high_prob_indices]
                y2_hp = self.data.desi_data['PMRA'].iloc[high_prob_indices]
                y3_hp = self.data.desi_data['PMDEC'].iloc[high_prob_indices]
                y4_hp = self.data.desi_data['FEH'].iloc[high_prob_indices]
                if residual_mode:
                    y1_hp = y1_hp - preds['desi'].get('vgsr', np.zeros(len(self.data.desi_data)))[high_prob_indices]
                    y2_hp = y2_hp - preds['desi'].get('pmra', np.zeros(len(self.data.desi_data)))[high_prob_indices]
                    y3_hp = y3_hp - preds['desi'].get('pmdec', np.zeros(len(self.data.desi_data)))[high_prob_indices]
                    y4_hp = y4_hp - preds['desi'].get('feh', np.zeros(len(self.data.desi_data)))[high_prob_indices]

                # Error bars for high-probability members (kinematics/abundance panels)
                # Ensure error bars are behind the colored member markers
                err_style = {'fmt': 'none'}
                err_style.update(self.plot_params.get('member_errorbar', {}))

                # Plot on each subplot
                ax[0].scatter(x_hp, self.data.desi_data[col_y0].iloc[high_prob_indices], **scatter_params)
                # Add error bars on panels 1-4 using respective measurement errors
                ax[1].errorbar(x_hp, y1_hp, yerr=self.data.desi_data['VRAD_ERR'].iloc[high_prob_indices], **err_style)
                ax[2].errorbar(x_hp, y2_hp, yerr=self.data.desi_data['PMRA_ERROR'].iloc[high_prob_indices], **err_style)
                ax[3].errorbar(x_hp, y3_hp, yerr=self.data.desi_data['PMDEC_ERROR'].iloc[high_prob_indices], **err_style)
                ax[4].errorbar(x_hp, y4_hp, yerr=self.data.desi_data['FEH_ERR'].iloc[high_prob_indices], **err_style)
                # Then scatter the colored member markers
                ax[1].scatter(x_hp, y1_hp, **{k: v for k, v in scatter_params.items() if k != 'label'})
                ax[2].scatter(x_hp, y2_hp, **{k: v for k, v in scatter_params.items() if k != 'label'})
                ax[3].scatter(x_hp, y3_hp, **{k: v for k, v in scatter_params.items() if k != 'label'})
                ax[4].scatter(x_hp, y4_hp, **{k: v for k, v in scatter_params.items() if k != 'label'})

                # Note: error bars for members are already plotted above via err_style
                
                # Add colorbar to the far right of all plots
                # Add colorbar to the overall figure instead of individual subplot
                # If ax is an array of Axes (like from plt.subplots)
                sm = cm.ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])

                # Ensure ax is a flat list of axes
                if isinstance(ax, np.ndarray):
                    ax = ax.ravel()

                # Position colorbar next to all subplots
                cbp = self.plot_params.get('membership_colorbar', {})
                cbar = fig.colorbar(sm, ax=ax,
                                    pad=cbp.get('pad', 0.02),
                                    aspect=cbp.get('aspect', 50),
                                    shrink=cbp.get('shrink', 1.0),
                                    location=cbp.get('location', 'right'))
                cbar.set_label(cbp.get('label', 'Membership Probability'), rotation=270, labelpad=cbp.get('labelpad', 15))
                
                # y-limits for residual mode will be set in a general block below (based on members)
            
            # Modify StreamFinder star styling when membership prob is shown
            if showStream:
                # Calculate membership probabilities for StreamFinder stars if possible
                sf_in_desi_indices = self.data.confirmed_sf_and_desi.index
                desi_indices = self.data.desi_data.index
                
                # Find which DESI indices correspond to SF stars
                sf_desi_mask = desi_indices.isin(sf_in_desi_indices)
                sf_prob_values = stream_prob[sf_desi_mask]
                
                # Determine colors for SF stars based on membership probability
                sf_high_prob_mask = sf_prob_values >= min_prob
                
                # Override existing StreamFinder star plots with new styling
                base_high = self.plot_params.get('sf_high_prob_diamond', {})
                sf_diamond_params_high = {**base_high,
                                          'c': sf_prob_values[sf_high_prob_mask],
                                          'cmap': cmap,
                                          'norm': norm}
                
                sf_diamond_params_low = self.plot_params.get('sf_low_prob_diamond', {}).copy()
                
                # Get high and low probability SF star indices
                sf_indices_high = sf_in_desi_indices[sf_high_prob_mask]
                sf_indices_low = sf_in_desi_indices[~sf_high_prob_mask]
                
                # Plot high probability SF stars with colormap
                if len(sf_indices_high) > 0:
                    sf_data_high = self.data.confirmed_sf_and_desi.loc[sf_indices_high]
                    
                    ax[0].scatter(
                        sf_data_high[col_x],
                        sf_data_high[col_y0],
                        **sf_diamond_params_high
                    )
                    ax[1].scatter(
                        sf_data_high[col_x],
                        (sf_data_high['VGSR'] - preds['sf'].get('vgsr', 0)[:len(sf_data_high)]) if residual_mode else sf_data_high['VGSR'],
                        **{k: v for k, v in sf_diamond_params_high.items() if k != 'label'}
                    )
                    ax[2].scatter(
                        sf_data_high[col_x],
                        (sf_data_high['PMRA'] - preds['sf'].get('pmra', 0)[:len(sf_data_high)]) if residual_mode else sf_data_high['PMRA'],
                        **{k: v for k, v in sf_diamond_params_high.items() if k != 'label'}
                    )
                    ax[3].scatter(
                        sf_data_high[col_x],
                        (sf_data_high['PMDEC'] - preds['sf'].get('pmdec', 0)[:len(sf_data_high)]) if residual_mode else sf_data_high['PMDEC'],
                        **{k: v for k, v in sf_diamond_params_high.items() if k != 'label'}
                    )
                    ax[4].scatter(
                        sf_data_high[col_x],
                        (sf_data_high['FEH'] - preds['sf'].get('feh', 0)[:len(sf_data_high)]) if residual_mode else sf_data_high['FEH'],
                        **{k: v for k, v in sf_diamond_params_high.items() if k != 'label'}
                    )
                
                # Plot low probability SF stars as black diamonds
                if len(sf_indices_low) > 0:
                    sf_data_low = self.data.confirmed_sf_and_desi.loc[sf_indices_low]
                    
                    ax[0].scatter(
                        sf_data_low[col_x],
                        sf_data_low[col_y0],
                        **sf_diamond_params_low,
                        label=f'SF Stars (<{min_prob:.1f})'
                    )
                    ax[1].scatter(
                        sf_data_low[col_x],
                        (sf_data_low['VGSR'] - preds['sf'].get('vgsr', 0)[-len(sf_data_low):]) if residual_mode else sf_data_low['VGSR'],
                        **sf_diamond_params_low
                    )
                    ax[2].scatter(
                        sf_data_low[col_x],
                        (sf_data_low['PMRA'] - preds['sf'].get('pmra', 0)[-len(sf_data_low):]) if residual_mode else sf_data_low['PMRA'],
                        **sf_diamond_params_low
                    )
                    ax[3].scatter(
                        sf_data_low[col_x],
                        (sf_data_low['PMDEC'] - preds['sf'].get('pmdec', 0)[-len(sf_data_low):]) if residual_mode else sf_data_low['PMDEC'],
                        **sf_diamond_params_low
                    )
                    ax[4].scatter(
                        sf_data_low[col_x],
                        (sf_data_low['FEH'] - preds['sf'].get('feh', 0)[-len(sf_data_low):]) if residual_mode else sf_data_low['FEH'],
                        **sf_diamond_params_low
                    )

        # Set consistent x-axis limits for all panels based on the data being plotted
        phi1_values_for_limits = []
        
        # Collect phi1 values from data sources that are actually being plotted
        if background and len(self.data.desi_data) > 0:
            phi1_values_for_limits.extend(self.data.desi_data[col_x].values)
        if showStream and hasattr(self.data, 'confirmed_sf_and_desi') and len(self.data.confirmed_sf_and_desi) > 0:
            phi1_values_for_limits.extend(self.data.confirmed_sf_and_desi[col_x].values)
        if show_sf_only and hasattr(self.data, 'confirmed_sf_not_desi') and len(self.data.confirmed_sf_not_desi) > 0:
            phi1_values_for_limits.extend(self.data.confirmed_sf_not_desi[col_x_].values)
        if hasattr(self.data, 'cut_confirmed_sf_and_desi') and show_cut and len(self.data.cut_confirmed_sf_and_desi) > 0:
            phi1_values_for_limits.extend(self.data.cut_confirmed_sf_and_desi[col_x].values)
        
        # Prefer member stars for x-limits if membership probabilities are available
        if show_membership_prob and stream_prob is not None:
            high_prob_mask = stream_prob >= min_prob
            high_prob_indices = np.where(high_prob_mask)[0]
            if len(high_prob_indices) > 0:
                # Get phi1 range from high probability member stars
                high_prob_phi1 = self.data.desi_data[col_x].iloc[high_prob_indices]
                phi1_min_members = high_prob_phi1.min()
                phi1_max_members = high_prob_phi1.max()
                
                # Add some padding
                phi1_range_members = phi1_max_members - phi1_min_members
                phi1_padding = 0.05 * phi1_range_members if phi1_range_members > 0 else 1.0
                x_limits = (phi1_min_members - phi1_padding, phi1_max_members + phi1_padding)
                
                # Apply consistent x-limits to all panels
                for panel_ax in ax:
                    panel_ax.set_xlim(x_limits)
        elif phi1_values_for_limits:
            # Use all plotted data for x-limits if no membership probabilities
            phi1_min_data = np.min(phi1_values_for_limits)
            phi1_max_data = np.max(phi1_values_for_limits)
            phi1_range_data = phi1_max_data - phi1_min_data
            phi1_padding = 0.02 * phi1_range_data if phi1_range_data > 0 else 1.0
            x_limits = (phi1_min_data - phi1_padding, phi1_max_data + phi1_padding)
            
            # Apply consistent x-limits to all panels
            for panel_ax in ax:
                panel_ax.set_xlim(x_limits)

            # In residuals mode, prefer y-limits based on high-probability member stars
            # Use member stars for limits whenever probabilities are available (independent of whether they are drawn)
            if residual_mode and stream_prob is not None:
                high_prob_mask = stream_prob >= min_prob
                high_prob_indices = np.where(high_prob_mask)[0]
                if len(high_prob_indices) > 0:
                    pad_v = self.plot_params.get('limits', {}).get('residual_pad_vgsr', 20)
                    pad_pm = self.plot_params.get('limits', {}).get('residual_pad_pm', 2)
                    pad_feh = self.plot_params.get('limits', {}).get('residual_pad_feh', 0.2)
                    # Build residual arrays from high-probability members
                    y1_hp = (self.data.desi_data['VGSR'].iloc[high_prob_indices]
                             - preds['desi'].get('vgsr', np.zeros(len(self.data.desi_data)))[high_prob_indices]).to_numpy()
                    y2_hp = (self.data.desi_data['PMRA'].iloc[high_prob_indices]
                             - preds['desi'].get('pmra', np.zeros(len(self.data.desi_data)))[high_prob_indices]).to_numpy()
                    y3_hp = (self.data.desi_data['PMDEC'].iloc[high_prob_indices]
                             - preds['desi'].get('pmdec', np.zeros(len(self.data.desi_data)))[high_prob_indices]).to_numpy()
                    # Set limits
                    ax[1].set_ylim(np.nanmin(y1_hp) - pad_v, np.nanmax(y1_hp) + pad_v)
                    ax[2].set_ylim(np.nanmin(y2_hp) - pad_pm, np.nanmax(y2_hp) + pad_pm)
                    ax[3].set_ylim(np.nanmin(y3_hp) - pad_pm, np.nanmax(y3_hp) + pad_pm)
                    if 'FEH' in self.data.desi_data.columns and isinstance(preds['desi'].get('feh', None), np.ndarray):
                        y4_hp = (self.data.desi_data['FEH'].iloc[high_prob_indices] - preds['desi']['feh'][high_prob_indices]).to_numpy()
                        ax[4].set_ylim(np.nanmin(y4_hp) - pad_feh, np.nanmax(y4_hp) + pad_feh)
                    else:
                        ax[4].set_ylim(-1, 1)

        # Set y-axis limits based on stream data if available (fallback for non-residuals)
        if not residual_mode:
            # Use membership probabilities to set y-limits if available
            if show_membership_prob and stream_prob is not None:
                high_prob_mask = stream_prob >= min_prob
                high_prob_indices = np.where(high_prob_mask)[0]
                if len(high_prob_indices) > 0:
                    # Get padding values from plot params
                    pad_v = self.plot_params.get('limits', {}).get('nonresidual_pad_vgsr', 50)
                    pad_pm = self.plot_params.get('limits', {}).get('nonresidual_pad_pm', 5)
                    pad_feh = self.plot_params.get('limits', {}).get('nonresidual_pad_feh', 0.5)
                    
                    # Get data from high-probability member stars
                    member_vgsr = self.data.desi_data['VGSR'].iloc[high_prob_indices]
                    member_pmra = self.data.desi_data['PMRA'].iloc[high_prob_indices]
                    member_pmdec = self.data.desi_data['PMDEC'].iloc[high_prob_indices]
                    
                    # Set limits based on member star ranges with padding
                    ax[1].set_ylim(np.nanmin(member_vgsr) - pad_v, np.nanmax(member_vgsr) + pad_v)
                    ax[2].set_ylim(np.nanmin(member_pmra) - pad_pm, np.nanmax(member_pmra) + pad_pm)
                    ax[3].set_ylim(np.nanmin(member_pmdec) - pad_pm, np.nanmax(member_pmdec) + pad_pm)
                    
                    # Handle metallicity if available
                    if 'FEH' in self.data.desi_data.columns:
                        member_feh = self.data.desi_data['FEH'].iloc[high_prob_indices]
                        if len(member_feh.dropna()) > 0:
                            ax[4].set_ylim(np.nanmin(member_feh) - pad_feh, np.nanmax(member_feh) + pad_feh)
                        else:
                            feh_ylim = self.plot_params.get('limits', {}).get('feh_ylim_default', (-4, -0.5))
                            ax[4].set_ylim(*feh_ylim)
                    else:
                        feh_ylim = self.plot_params.get('limits', {}).get('feh_ylim_default', (-4, -0.5))
                        ax[4].set_ylim(*feh_ylim)
            
            # Fallback to StreamFinder data if no membership probabilities but showStream is True
            elif showStream and hasattr(self.data, 'confirmed_sf_and_desi') and len(self.data.confirmed_sf_and_desi) > 0:
                # VGSR limits
                vgsr_data = [self.data.confirmed_sf_and_desi['VGSR']]
                if hasattr(self.data, 'cut_confirmed_sf_and_desi') and len(self.data.cut_confirmed_sf_and_desi) > 0 and show_cut:
                    vgsr_data.append(self.data.cut_confirmed_sf_and_desi['VGSR'])
                vgsr_combined = np.concatenate([np.array(x) for x in vgsr_data])
                ax[1].set_ylim(np.nanmin(vgsr_combined) - 50, np.nanmax(vgsr_combined) + 50)
                
                # Proper motion limits  
                pmra_data = [self.data.confirmed_sf_and_desi['PMRA']]
                pmdec_data = [self.data.confirmed_sf_and_desi['PMDEC']]
                if hasattr(self.data, 'cut_confirmed_sf_and_desi') and len(self.data.cut_confirmed_sf_and_desi) > 0 and show_cut:
                    pmra_data.append(self.data.cut_confirmed_sf_and_desi['PMRA'])
                    pmdec_data.append(self.data.cut_confirmed_sf_and_desi['PMDEC'])
                pmra_combined = np.concatenate([np.array(x) for x in pmra_data])
                pmdec_combined = np.concatenate([np.array(x) for x in pmdec_data])
                ax[2].set_ylim(np.nanmin(pmra_combined) - 5, np.nanmax(pmra_combined) + 5)
                ax[3].set_ylim(np.nanmin(pmdec_combined) - 5, np.nanmax(pmdec_combined) + 5)

                # Set metallicity limits (non-residuals default)
                feh_ylim = self.plot_params.get('limits', {}).get('feh_ylim_default', (-4, -0.5))
                ax[4].set_ylim(*feh_ylim)
        
        # Plot splines if requested and available
        if (show_initial_splines or show_optimized_splines or show_mcmc_splines) and stream_frame and self.mcmeta is not None:
            # Create phi1 range for spline plotting based on data range, not current axis limits
            phi1_values = []
            
            # Collect phi1 values from all data sources that will be plotted
            if background and len(self.data.desi_data) > 0:
                phi1_values.extend(self.data.desi_data[col_x].values)
            if showStream and hasattr(self.data, 'confirmed_sf_and_desi') and len(self.data.confirmed_sf_and_desi) > 0:
                phi1_values.extend(self.data.confirmed_sf_and_desi[col_x].values)
            if show_sf_only and hasattr(self.data, 'confirmed_sf_not_desi') and len(self.data.confirmed_sf_not_desi) > 0:
                phi1_values.extend(self.data.confirmed_sf_not_desi[col_x_].values)
            if hasattr(self.data, 'cut_confirmed_sf_and_desi') and show_cut and len(self.data.cut_confirmed_sf_and_desi) > 0:
                phi1_values.extend(self.data.cut_confirmed_sf_and_desi[col_x].values)
            
            # Also include membership probability data if it's being shown
            if show_membership_prob and stream_prob is not None:
                # Always include the full DESI dataset range when showing membership probabilities
                phi1_values.extend(self.data.desi_data[col_x].values)
            
            # If we have phi1 values, use them to determine the range, otherwise fall back to axis limits
            if phi1_values:
                phi1_min = np.min(phi1_values)
                phi1_max = np.max(phi1_values)
                # Add some padding to ensure we cover the full range
                phi1_range = phi1_max - phi1_min
                phi1_min -= 0.1 * phi1_range
                phi1_max += 0.1 * phi1_range
            else:
                # Fallback to current axis limits if no data available
                phi1_min = ax[1].get_xlim()[0]
                phi1_max = ax[1].get_xlim()[1]
            
            phi1_spline_plot = np.linspace(phi1_min, phi1_max, 200)  # Increased resolution
            
        # If optimized splines are requested but optimized_params aren't attached,
        # try to source them from common places used in the notebook (silently).
            if show_optimized_splines and (not hasattr(self.mcmeta, 'optimized_params') or self.mcmeta.optimized_params is None):
                try:
                    import inspect
                    sourced = False
                    # 1) Look for a variable named `optimized_for_plotting` in caller frames
                    frame = inspect.currentframe()
                    caller_frame = frame.f_back if frame is not None else None
                    while caller_frame:
                        if 'optimized_for_plotting' in caller_frame.f_globals:
                            ofp = caller_frame.f_globals['optimized_for_plotting']
                            if ofp is not None:
                                self.mcmeta.optimized_params = ofp
                                sourced = True
                                break
                        caller_frame = caller_frame.f_back
                    # 2) Fallback to MCMeta.sp_output (from scipy_optimize)
                    if not sourced and hasattr(self.mcmeta, 'sp_output') and self.mcmeta.sp_output is not None:
                        self.mcmeta.optimized_params = self.mcmeta.sp_output
                        sourced = True
                except Exception as e:
                    pass
            
            # Plot initial guess splines in black
            if show_initial_splines:
                if hasattr(self.mcmeta, 'phi1_spline_points'):
                    try:
                        # VGSR spline
                        vgsr_initial = stream_funcs.apply_spline(
                            phi1_spline_plot, self.mcmeta.phi1_spline_points, 
                            self.mcmeta.initial_params['vgsr_spline_points'], k=2
                        )
                        if 'vgsr_spline_points' in self.mcmeta.initial_params and 'vgsr' in locals():
                            pass
                        # Residual vs MCMC if applicable
                        if 'vgsr_knots' in locals() and 'residual_mode' in locals() and residual_mode:
                            vgsr_mcmc_eval = stream_funcs.apply_spline(
                                phi1_spline_plot, self.mcmeta.phi1_spline_points, vgsr_knots, k=2
                            )
                            vgsr_initial_plot = vgsr_initial - vgsr_mcmc_eval
                        else:
                            vgsr_initial_plot = vgsr_initial
                        ax[1].plot(
                            phi1_spline_plot,
                            vgsr_initial_plot,
                            **self.plot_params.get('initial_spline_line', {})
                        )
                        
                        # Add circle markers at spline points
                        if 'vgsr_knots' in locals() and residual_mode:
                            vgsr_mcmc_knots = stream_funcs.apply_spline(
                                self.mcmeta.phi1_spline_points, self.mcmeta.phi1_spline_points, vgsr_knots, k=2
                            )
                            vgsr_init_knots_scatter = self.mcmeta.initial_params['vgsr_spline_points'] - vgsr_mcmc_knots
                        else:
                            vgsr_init_knots_scatter = self.mcmeta.initial_params['vgsr_spline_points']
                        ax[1].scatter(
                            self.mcmeta.phi1_spline_points,
                            vgsr_init_knots_scatter,
                            **self.plot_params.get('spline_knot_initial', {})
                        )
                        
                        # PMRA spline
                        pmra_initial = stream_funcs.apply_spline(
                            phi1_spline_plot, self.mcmeta.phi1_spline_points, 
                            self.mcmeta.initial_params['pmra_spline_points'], k=2
                        )
                        if 'pmra_knots' in locals() and 'residual_mode' in locals() and residual_mode:
                            pmra_mcmc_eval = stream_funcs.apply_spline(
                                phi1_spline_plot, self.mcmeta.phi1_spline_points, pmra_knots, k=2
                            )
                            pmra_initial_plot = pmra_initial - pmra_mcmc_eval
                        else:
                            pmra_initial_plot = pmra_initial
                        ax[2].plot(
                            phi1_spline_plot,
                            pmra_initial_plot,
                            **self.plot_params.get('initial_spline_line', {})
                        )
                        
                        # Add circle markers at spline points
                        if 'pmra_knots' in locals() and residual_mode:
                            pmra_mcmc_knots = stream_funcs.apply_spline(
                                self.mcmeta.phi1_spline_points, self.mcmeta.phi1_spline_points, pmra_knots, k=2
                            )
                            pmra_init_knots_scatter = self.mcmeta.initial_params['pmra_spline_points'] - pmra_mcmc_knots
                        else:
                            pmra_init_knots_scatter = self.mcmeta.initial_params['pmra_spline_points']
                        ax[2].scatter(
                            self.mcmeta.phi1_spline_points,
                            pmra_init_knots_scatter,
                            **self.plot_params.get('spline_knot_initial', {})
                        )
                        
                        # PMDEC spline
                        pmdec_initial = stream_funcs.apply_spline(
                            phi1_spline_plot, self.mcmeta.phi1_spline_points, 
                            self.mcmeta.initial_params['pmdec_spline_points'], k=2
                        )
                        if 'pmdec_knots' in locals() and 'residual_mode' in locals() and residual_mode:
                            pmdec_mcmc_eval = stream_funcs.apply_spline(
                                phi1_spline_plot, self.mcmeta.phi1_spline_points, pmdec_knots, k=2
                            )
                            pmdec_initial_plot = pmdec_initial - pmdec_mcmc_eval
                        else:
                            pmdec_initial_plot = pmdec_initial
                        ax[3].plot(
                            phi1_spline_plot,
                            pmdec_initial_plot,
                            **self.plot_params.get('initial_spline_line', {})
                        )
                        
                        # Add circle markers at spline points
                        if 'pmdec_knots' in locals() and residual_mode:
                            pmdec_mcmc_knots = stream_funcs.apply_spline(
                                self.mcmeta.phi1_spline_points, self.mcmeta.phi1_spline_points, pmdec_knots, k=2
                            )
                            pmdec_init_knots_scatter = self.mcmeta.initial_params['pmdec_spline_points'] - pmdec_mcmc_knots
                        else:
                            pmdec_init_knots_scatter = self.mcmeta.initial_params['pmdec_spline_points']
                        ax[3].scatter(
                            self.mcmeta.phi1_spline_points,
                            pmdec_init_knots_scatter,
                            **self.plot_params.get('spline_knot_initial', {})
                        )
                        
                        # FEH constant line
                        feh_initial = np.full_like(phi1_spline_plot, self.mcmeta.initial_params['feh1'])
                        if 'feh_const' in locals() and 'residual_mode' in locals() and residual_mode:
                            feh_initial_plot = feh_initial - feh_const
                        else:
                            feh_initial_plot = feh_initial
                        ax[4].plot(
                            phi1_spline_plot,
                            feh_initial_plot,
                            **self.plot_params.get('initial_feh_line', {})
                        )
                        
                        # No FEH spline markers on metallicity panel (line only)
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
                        if 'vgsr_knots' in locals() and 'residual_mode' in locals() and residual_mode:
                            vgsr_mcmc_eval = stream_funcs.apply_spline(
                                phi1_spline_plot, self.mcmeta.phi1_spline_points, vgsr_knots, k=2
                            )
                            vgsr_optimized_plot = vgsr_optimized - vgsr_mcmc_eval
                        else:
                            vgsr_optimized_plot = vgsr_optimized
                        ax[1].plot(
                            phi1_spline_plot,
                            vgsr_optimized_plot,
                            **self.plot_params.get('optimized_spline_line', {})
                        )
                        
                        # Add circle markers at spline points
                        if 'vgsr_knots' in locals() and 'residual_mode' in locals() and residual_mode:
                            vgsr_mcmc_knots = stream_funcs.apply_spline(
                                self.mcmeta.phi1_spline_points, self.mcmeta.phi1_spline_points, vgsr_knots, k=2
                            )
                            vgsr_knots_scatter = self.mcmeta.optimized_params['vgsr_spline_points'] - vgsr_mcmc_knots
                        else:
                            vgsr_knots_scatter = self.mcmeta.optimized_params['vgsr_spline_points']
                        ax[1].scatter(
                            self.mcmeta.phi1_spline_points,
                            vgsr_knots_scatter,
                            **self.plot_params.get('spline_knot_optimized', {})
                        )
                        
                        # PMRA spline
                        pmra_optimized = stream_funcs.apply_spline(
                            phi1_spline_plot, self.mcmeta.phi1_spline_points, 
                            self.mcmeta.optimized_params['pmra_spline_points'], k=2
                        )
                        if 'pmra_knots' in locals() and 'residual_mode' in locals() and residual_mode:
                            pmra_mcmc_eval = stream_funcs.apply_spline(
                                phi1_spline_plot, self.mcmeta.phi1_spline_points, pmra_knots, k=2
                            )
                            pmra_optimized_plot = pmra_optimized - pmra_mcmc_eval
                        else:
                            pmra_optimized_plot = pmra_optimized
                        ax[2].plot(
                            phi1_spline_plot,
                            pmra_optimized_plot,
                            **self.plot_params.get('optimized_spline_line', {})
                        )
                        
                        # Add circle markers at spline points
                        if 'pmra_knots' in locals() and 'residual_mode' in locals() and residual_mode:
                            pmra_mcmc_knots = stream_funcs.apply_spline(
                                self.mcmeta.phi1_spline_points, self.mcmeta.phi1_spline_points, pmra_knots, k=2
                            )
                            pmra_knots_scatter = self.mcmeta.optimized_params['pmra_spline_points'] - pmra_mcmc_knots
                        else:
                            pmra_knots_scatter = self.mcmeta.optimized_params['pmra_spline_points']
                        ax[2].scatter(
                            self.mcmeta.phi1_spline_points,
                            pmra_knots_scatter,
                            **self.plot_params.get('spline_knot_optimized', {})
                        )
                        
                        # PMDEC spline
                        pmdec_optimized = stream_funcs.apply_spline(
                            phi1_spline_plot, self.mcmeta.phi1_spline_points, 
                            self.mcmeta.optimized_params['pmdec_spline_points'], k=2
                        )
                        if 'pmdec_knots' in locals() and 'residual_mode' in locals() and residual_mode:
                            pmdec_mcmc_eval = stream_funcs.apply_spline(
                                phi1_spline_plot, self.mcmeta.phi1_spline_points, pmdec_knots, k=2
                            )
                            pmdec_optimized_plot = pmdec_optimized - pmdec_mcmc_eval
                        else:
                            pmdec_optimized_plot = pmdec_optimized
                        ax[3].plot(
                            phi1_spline_plot,
                            pmdec_optimized_plot,
                            **self.plot_params.get('optimized_spline_line', {})
                        )
                        
                        # Add circle markers at spline points
                        if 'pmdec_knots' in locals() and 'residual_mode' in locals() and residual_mode:
                            pmdec_mcmc_knots = stream_funcs.apply_spline(
                                self.mcmeta.phi1_spline_points, self.mcmeta.phi1_spline_points, pmdec_knots, k=2
                            )
                            pmdec_knots_scatter = self.mcmeta.optimized_params['pmdec_spline_points'] - pmdec_mcmc_knots
                        else:
                            pmdec_knots_scatter = self.mcmeta.optimized_params['pmdec_spline_points']
                        ax[3].scatter(
                            self.mcmeta.phi1_spline_points,
                            pmdec_knots_scatter,
                            **self.plot_params.get('spline_knot_optimized', {})
                        )
                        
                        # FEH constant line
                        feh_optimized = np.full_like(phi1_spline_plot, self.mcmeta.optimized_params['feh1'])
                        if 'feh_const' in locals() and 'residual_mode' in locals() and residual_mode:
                            feh_optimized_plot = feh_optimized - feh_const
                        else:
                            feh_optimized_plot = feh_optimized
                        ax[4].plot(
                            phi1_spline_plot,
                            feh_optimized_plot,
                            **self.plot_params.get('optimized_feh_line', {})
                        )
                        
                        # No FEH spline markers on metallicity panel (line only)
                    except Exception as e:
                        print(f"Warning: Could not plot optimized splines: {e}")
            
            # Plot MCMC splines in blue (requires external meds dictionary with MCMC results)
            if show_mcmc_splines and hasattr(self.mcmeta, 'phi1_spline_points'):
                        
                            
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
                        if residual_mode:
                            ax[1].axhline(0, color='blue', linewidth=2, alpha=0.8, label='MCMC Spline (resid)')
                        else:
                            ax[1].plot(phi1_spline_plot, vgsr_mcmc, 'b-', linewidth=2, 
                                      label='MCMC Spline', alpha=0.8)
                        
                        # Plot 1 sigma and 2 sigma regions covering full spline length
                        sigma_vgsr = 10**meds['lsigvgsr']
                        print(phi1_spline_plot.min())
                        if residual_mode:
                            ax[1].fill_between(phi1_spline_plot, -sigma_vgsr, +sigma_vgsr, color='blue', alpha=0.1)
                            ax[1].fill_between(phi1_spline_plot, -2*sigma_vgsr, +2*sigma_vgsr, color='blue', alpha=0.05)
                        else:
                            ax[1].fill_between(phi1_spline_plot, vgsr_mcmc - sigma_vgsr, vgsr_mcmc + sigma_vgsr, color='blue', alpha=0.1)
                            ax[1].fill_between(phi1_spline_plot, vgsr_mcmc - 2*sigma_vgsr, vgsr_mcmc + 2*sigma_vgsr, color='blue', alpha=0.05)

                        vgsr_mcmc_ep = []
                        for i in range(1, no_of_spline_points + 1):
                            vgsr_mcmc_ep.append(ep['vgsr'+str(i)])

                        vgsr_mcmc_em = []
                        for i in range(1, no_of_spline_points + 1):
                            vgsr_mcmc_em.append(np.abs(em['vgsr'+str(i)]))
                        
                        vgsr_mcmc_errors = np.array([vgsr_mcmc_em, vgsr_mcmc_ep])
                        yvals = np.zeros_like(self.mcmeta.phi1_spline_points) if residual_mode else vgsr_mcmc_points
                        ax[1].errorbar(self.mcmeta.phi1_spline_points, yvals, 
                                        yerr=vgsr_mcmc_errors, fmt='o', color='blue', 
                                        markersize=8, zorder=10, alpha=0.8, markeredgecolor='white', 
                                        markeredgewidth=1, capsize=3, capthick=1.5, elinewidth=1.5)

                        # PMRA spline
                        pmra_mcmc = stream_funcs.apply_spline(
                            phi1_spline_plot, self.mcmeta.phi1_spline_points, 
                            pmra_mcmc_points, k=2
                        )
                        if residual_mode:
                            ax[2].axhline(0, color='blue', linewidth=2, alpha=0.8, label='MCMC Spline (resid)')
                        else:
                            ax[2].plot(phi1_spline_plot, pmra_mcmc, 'b-', linewidth=2, 
                                      label='MCMC Spline', alpha=0.8)
                        
                        # Plot 1 sigma and 2 sigma regions for PMRA
                        sigma_pmra = 10**meds['lsigpmra']
                        if residual_mode:
                            ax[2].fill_between(phi1_spline_plot, -sigma_pmra, +sigma_pmra, color='blue', alpha=0.1)
                            ax[2].fill_between(phi1_spline_plot, -2*sigma_pmra, +2*sigma_pmra, color='blue', alpha=0.05)
                        else:
                            ax[2].fill_between(phi1_spline_plot, pmra_mcmc - sigma_pmra, pmra_mcmc + sigma_pmra, color='blue', alpha=0.1)
                            ax[2].fill_between(phi1_spline_plot, pmra_mcmc - 2*sigma_pmra, pmra_mcmc + 2*sigma_pmra, color='blue', alpha=0.05)

                        # Extract error bars for PMRA spline points
                        pmra_mcmc_ep = []
                        for i in range(1, no_of_spline_points + 1):
                            pmra_mcmc_ep.append(ep['pmra'+str(i)])

                        pmra_mcmc_em = []
                        for i in range(1, no_of_spline_points + 1):
                            pmra_mcmc_em.append(np.abs(em['pmra'+str(i)]))
                        pmra_mcmc_errors = np.array([pmra_mcmc_em, pmra_mcmc_ep])

                        yvals = np.zeros_like(self.mcmeta.phi1_spline_points) if residual_mode else pmra_mcmc_points
                        ax[2].errorbar(self.mcmeta.phi1_spline_points, yvals, 
                                        yerr=pmra_mcmc_errors, fmt='o', color='blue', 
                                        markersize=8, zorder=10, alpha=0.8, markeredgecolor='white', 
                                        markeredgewidth=1, capsize=3, capthick=1.5, elinewidth=1.5)

                        
                        # PMDEC spline
                        pmdec_mcmc = stream_funcs.apply_spline(
                            phi1_spline_plot, self.mcmeta.phi1_spline_points, 
                            pmdec_mcmc_points, k=2
                        )
                        if residual_mode:
                            ax[3].axhline(0, color='blue', linewidth=2, alpha=0.8, label='MCMC Spline (resid)')
                        else:
                            ax[3].plot(phi1_spline_plot, pmdec_mcmc, 'b-', linewidth=2, 
                                      label='MCMC Spline', alpha=0.8)
                        
                        # Plot 1 sigma and 2 sigma regions for PMDEC
                        sigma_pmdec = 10**meds['lsigpmdec']
                        if residual_mode:
                            ax[3].fill_between(phi1_spline_plot, -sigma_pmdec, +sigma_pmdec, color='blue', alpha=0.1)
                            ax[3].fill_between(phi1_spline_plot, -2*sigma_pmdec, +2*sigma_pmdec, color='blue', alpha=0.05)
                        else:
                            ax[3].fill_between(phi1_spline_plot, pmdec_mcmc - sigma_pmdec, pmdec_mcmc + sigma_pmdec, color='blue', alpha=0.1)
                            ax[3].fill_between(phi1_spline_plot, pmdec_mcmc - 2*sigma_pmdec, pmdec_mcmc + 2*sigma_pmdec, color='blue', alpha=0.05)
                        
                        # Extract error bars for PMDEC spline points


                        pmdec_mcmc_ep = []
                        for i in range(1, no_of_spline_points + 1):
                            pmdec_mcmc_ep.append(ep['pmdec'+str(i)])

                        pmdec_mcmc_em = []
                        for i in range(1, no_of_spline_points + 1):
                            pmdec_mcmc_em.append(np.abs(em['pmdec'+str(i)]))
                        pmdec_mcmc_errors = np.array([pmdec_mcmc_em, pmdec_mcmc_ep])

                        yvals = np.zeros_like(self.mcmeta.phi1_spline_points) if residual_mode else pmdec_mcmc_points
                        ax[3].errorbar(self.mcmeta.phi1_spline_points, yvals, 
                                        yerr=pmdec_mcmc_errors, fmt='o', color='blue', 
                                        markersize=8, zorder=10, alpha=0.8, markeredgecolor='white', 
                                        markeredgewidth=1, capsize=3, capthick=1.5, elinewidth=1.5)

                        # FEH constant line
                        feh_mcmc = np.full_like(phi1_spline_plot, meds['feh1'])
                        if residual_mode:
                            ax[4].axhline(0, color='blue', linewidth=2, alpha=0.8, label='MCMC [Fe/H] (resid)')
                        else:
                            ax[4].plot(phi1_spline_plot, feh_mcmc, 'b-', linewidth=2, 
                                      label='MCMC [Fe/H]', alpha=0.8)
                        
                        # Plot 1 sigma and 2 sigma regions for FEH
                        sigma_feh = 10**meds['lsigfeh']
                        if residual_mode:
                            ax[4].fill_between(phi1_spline_plot, -sigma_feh, +sigma_feh, color='blue', alpha=0.1)
                            ax[4].fill_between(phi1_spline_plot, -2*sigma_feh, +2*sigma_feh, color='blue', alpha=0.05)
                        else:
                            ax[4].fill_between(phi1_spline_plot, feh_mcmc - sigma_feh, feh_mcmc + sigma_feh, color='blue', alpha=0.1)
                            ax[4].fill_between(phi1_spline_plot, feh_mcmc - 2*sigma_feh, feh_mcmc + 2*sigma_feh, color='blue', alpha=0.05)
                        
                        
        
        # Labels and formatting
        if show_initial_splines or show_optimized_splines or show_mcmc_splines:
            # Add legends to kinematic plots if splines are shown
            for i in [1]:  # Show legends on all kinematic plots
                if ax[i].get_lines():  # Only add legend if there are lines to show
                    ax[i].legend(loc='best', fontsize='small')
        
        ax[0].legend(loc='upper left', ncol=4)
        ax[0].set_ylabel(label_y0)
        ax[1].set_ylabel(r'V$_{GSR}$ (km/s)' if not residual_mode else r'V$_{GSR}$ - v$_{\rm MCMC}$ (km/s)')
        ax[2].set_ylabel(r'$\mu_{\alpha}$ [mas/yr]' if not residual_mode else r'$\mu_{\alpha}$ - $\mu_{\alpha,\rm MCMC}$ [mas/yr]')
        ax[3].set_ylabel(r'$\mu_{\delta}$ [mas/yr]' if not residual_mode else r'$\mu_{\delta}$ - $\mu_{\delta,\rm MCMC}$ [mas/yr]')
        ax[4].set_ylabel(r'[Fe/H]' if not residual_mode else r'[Fe/H] - [Fe/H]$_{\rm MCMC}$')
        ax[-1].set_xlabel(label_x)
        
        for a in ax:
            stream_funcs.plot_form(a)
            
        if save:
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}sixD_plot_{self.stream.streamName}.png", dpi=300, bbox_inches='tight')
            
        return fig, ax
    
    def gaussian_mixture_plot(self, showStream=True, background=True, save=False, show_model=True, show_total=True):
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
        
        if show_model:
            ax.plot(vgsr_range, stream_weight * stream_vgsr_pdf, ':', color=colors[0], label='Stream Component', lw=3, zorder=2)
            ax.plot(vgsr_range, bg_weight * bg_vgsr_pdf, ':', color=colors[1], label='Background Component', lw=3, zorder=2)
        if show_model and show_total:
            ax.plot(vgsr_range, stream_weight * stream_vgsr_pdf + bg_weight * bg_vgsr_pdf, 'k-', label='Total Model', lw=3,zorder=1)
            
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
        if show_model:
            ax.plot(feh_range, stream_weight * stream_feh_pdf, ':', color=colors[0], lw=3, zorder=2)
            ax.plot(feh_range, bg_weight * bg_feh_pdf, ':', color=colors[1], lw=3, zorder=2)
        if show_model and show_total:
            ax.plot(feh_range, stream_weight * stream_feh_pdf + bg_weight * bg_feh_pdf, 'k-', lw=3, zorder=1)
            
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
        if show_model:
            ax.plot(pmra_range, stream_weight * stream_pmra_pdf, ':', color=colors[0], lw=3, zorder=2)
            ax.plot(pmra_range, bg_weight * bg_pmra_pdf, ':', color=colors[1], lw=3, zorder=2)
        if show_model and show_total:
            ax.plot(pmra_range, stream_weight * stream_pmra_pdf + bg_weight * bg_pmra_pdf, 'k-', lw=3, zorder=1)
            
 
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
        if show_model:
            ax.plot(pmdec_range, stream_weight * stream_pmdec_pdf, ':', color=colors[0], lw=3, zorder=2)
            ax.plot(pmdec_range, bg_weight * bg_pmdec_pdf, ':', color=colors[1], lw=3, zorder=2)
        if show_model and show_total:
            ax.plot(pmdec_range, stream_weight * stream_pmdec_pdf + bg_weight * bg_pmdec_pdf, 'k-', lw=3, zorder=1)
        
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
    def __init__(self, no_of_spline_points, stream_object, sf_data, truncation_params=None, phi1_min=None, phi1_max=None):
        self.stream = stream_object
        self.no_of_spline_points = no_of_spline_points
        self.sf_data = sf_data
        if self.no_of_spline_points == 1:
            self.spline_k = 1
        elif self.no_of_spline_points > 3:
            self.spline_k = 3
        else:
            self.spline_k = self.no_of_spline_points - 1

        # Use provided phi1 range or default to StreamFinder data range
        if phi1_min is None:
            phi1_min = self.stream.data.SoI_streamfinder['phi1'].min()
        if phi1_max is None:
            phi1_max = self.stream.data.SoI_streamfinder['phi1'].max()
        
        self.phi1_spline_points = np.linspace(phi1_min, phi1_max, self.no_of_spline_points)

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
    
    def priors(self, prior_arr):
        self.prior_arr = prior_arr

        self.p0_guess = [
        0.1,                                                 # pstream (constant stream fraction)
        self.initial_params['vgsr_spline_points'],         # VGSR spline points
        self.initial_params['lsigvgsr'],                   # lsigvgsr (constant log velocity dispersion)
        self.initial_params['feh1'],                       # mean [Fe/H]
        self.initial_params['lsigfeh'],                    # log(sigma_[Fe/H])
        self.initial_params['pmra_spline_points'],         # PMRA spline points
        self.initial_params['lsigpmra'],                   # log(sigma_pmra)
        self.initial_params['pmdec_spline_points'],        # PMDEC spline points
        self.initial_params['lsigpmdec'],                  # log(sigma_pmdec)
        self.initial_params['bv'],                         # background VGSR
        self.initial_params['lsigbv'],                     # log(sigma_background_vgsr)
        self.initial_params['bfeh'],                       # background [Fe/H]
        self.initial_params['lsigbfeh'],                   # log(sigma_background_feh)
        self.initial_params['bpmra'],                      # background PMRA
        self.initial_params['lsigbpmra'],                  # log(sigma_background_pmra)
        self.initial_params['bpmdec'],                     # background PMDEC
        self.initial_params['lsigbpmdec']                  # log(sigma_background_pmdec)
    ]
        
        self.vgsr_trunc = [self.truncation_params['vgsr_min'], self.truncation_params['vgsr_max']]
        self.feh_trunc = [self.truncation_params['feh_min'], self.truncation_params['feh_max']]  
        self.pmra_trunc = [self.truncation_params['pmra_min'], self.truncation_params['pmra_max']]
        self.pmdec_trunc = [self.truncation_params['pmdec_min'], self.truncation_params['pmdec_max']]

        self.array_lengths = [len(x) if isinstance(x, np.ndarray) else 1 for x in self.p0_guess]
        self.flat_p0_guess = np.hstack(self.p0_guess) 

    def scipy_optimize(self):
        self.param_labels = ['pstream', 'vgsr_spline_points', 'lsigvgsr', 'feh1', 'lsigfeh', 
            'pmra_spline_points', 'lsigpmra', 'pmdec_spline_points', 'lsigpmdec',
                'bv', 'lsigbv', 'bfeh', 'lsigbfeh', 'bpmra', 'lsigbpmra', 'bpmdec', 'lsigbpmdec']
        optfunc = lambda theta: -stream_funcs.spline_lnprob_1D(
            theta, self.prior_arr, self.phi1_spline_points,  # Only phi1_spline_points needed
            self.stream.data.desi_data['VGSR'], self.stream.data.desi_data['VRAD_ERR'],
            self.stream.data.desi_data['FEH'], self.stream.data.desi_data['FEH_ERR'],
            self.stream.data.desi_data['PMRA'], self.stream.data.desi_data['PMRA_ERROR'],
            self.stream.data.desi_data['PMDEC'], self.stream.data.desi_data['PMDEC_ERROR'],
            self.stream.data.desi_data['phi1'], 
            trunc_fit=True, feh_fit=True, assert_prior=False, k=self.spline_k, 
            reshape_arr_shape=self.array_lengths,
            vgsr_trunc=self.vgsr_trunc, feh_trunc=self.feh_trunc, 
            pmra_trunc=self.pmra_trunc, pmdec_trunc=self.pmdec_trunc
        )
        # Run optimization
        print("Running optimization...")
        self.sp_result = sp.optimize.minimize(optfunc, self.flat_p0_guess, method="Nelder-Mead")
        print(self.sp_result.message)

        self.reshaped_result = stream_funcs.reshape_arr(self.sp_result.x, self.array_lengths)
        self.sp_output = stream_funcs.get_paramdict(self.reshaped_result, labels=self.param_labels)
        # Make optimized parameters available for plotting/debugging
        self.optimized_params = self.sp_output

        print("\nOptimized Parameters:")
        for label, value in self.sp_output.items():
            if label.startswith('l'):
                if isinstance(value, np.ndarray):
                    print(f"{label[1:]}: {10**value}")
                else:
                    print(f"{label[1:]}: {10**value:.4f}")
            else:
                if isinstance(value, np.ndarray):
                    print(f"{label}: {value}")
                else:
                    print(f"{label}: {value:.4f}" if isinstance(value, (int, float)) else f"{label}: {value}")

    def prior_validation(self):
        self.nparams = len(self.param_labels)
        self.nwalkers = 70

        self.p0 = self.flat_p0_guess 
        self.ep0 = np.zeros(len(self.p0)) + 0.01

        # Generate walker positions around the starting point
        p0s = np.random.multivariate_normal(self.p0, np.diag(self.ep0)**2, size=self.nwalkers)

        # Clip pstream to valid range [0, 1] - first parameter only
        p0s[:,0] = np.clip(p0s[:,0], 1e-10, 1 - 1e-10)

        # Test likelihood for all walkers using the modified function
        lkhds = [stream_funcs.spline_lnprob_1D(
            p0s[j], self.prior_arr, self.phi1_spline_points, 
            self.stream.data.desi_data['VGSR'], self.stream.data.desi_data['VRAD_ERR'],
            self.stream.data.desi_data['FEH'], self.stream.data.desi_data['FEH_ERR'],
            self.stream.data.desi_data['PMRA'], self.stream.data.desi_data['PMRA_ERROR'],
            self.stream.data.desi_data['PMDEC'], self.stream.data.desi_data['PMDEC_ERROR'],
            self.stream.data.desi_data['phi1'], 
            trunc_fit=True, feh_fit=True, assert_prior=True, k=self.spline_k, 
            reshape_arr_shape=self.array_lengths,
            vgsr_trunc=self.vgsr_trunc, feh_trunc=self.feh_trunc, 
            pmra_trunc=self.pmra_trunc, pmdec_trunc=self.pmdec_trunc
        ) for j in range(self.nwalkers)]

        # Check if prior is good - this is the key test from your original code
        if sum(np.array(lkhds) > -9e9) == self.nwalkers:
            print('Your prior is good, you\'ve found something!')
        elif sum(np.array(lkhds) > -9e9) != self.nwalkers:
            print('Your prior is too restrictive, try changing the values listed above!')

        # Assert that all walkers have good likelihoods
        assert np.all(np.array(lkhds) > -9e9), f"Only {sum(np.array(lkhds) > -9e9)}/{self.nwalkers} walkers have valid likelihoods"

        print(f"All {self.nwalkers} walkers initialized successfully!")


    

class MCMC:
    """
    For running MCMC and intial outputs
    """
    def __init__(self, MCMeta_object, output_dir=''):
        
        self.meta = MCMeta_object
        self.stream = self.meta.stream
        self.output_dir = output_dir
        self.backend = emcee.backends.HDFBackend(self.output_dir+'/'+self.stream.streamName+str(self.meta.no_of_spline_points)+'.h5')
        self.backend.reset(self.meta.nwalkers,len(self.meta.p0))
    #WIP
    def run(self, nproc=32, nburnin=5000, nstep=5000, use_optimized_start=True):
        from multiprocessing import Pool
        self.nproc = nproc
        self.nburnin = nburnin
        self.nstep = nstep
        self.use_optimized_start = use_optimized_start
        if self.use_optimized_start:
            print("Using optimized parameters as starting positions...")
            start_params = self.meta.sp_result.x
            start_label = "optimized"
        else:
            print("Using initial guess as starting positions...")
            start_params = self.meta.flat_p0_guess
            start_label = "initial_guess"

        with Pool(self.nproc) as pool:
            print(f"Running burn-in with {self.nburnin} iterations. starting from {start_label} parameters...")
            p0 = start_params
            ep0 = np.zeros(len(p0)) + 0.01
            assert np.all(np.isfinite(start_params)), "start_params contains NaN or inf"
            # Generate walker positions around the starting point
            p0s = np.random.multivariate_normal(p0, np.diag(ep0)**2, size=self.meta.nwalkers)

            print("Clipping all walker positions to be within prior ranges...")
            for i in range(len(self.meta.prior_arr)):
                min_val, max_val = self.meta.prior_arr[i]
                # Add a small buffer to avoid being exactly on the boundary
                buffer = 1e-10
                p0s[:, i] = np.clip(p0s[:, i], min_val + buffer, max_val - buffer)

            # Special clipping for pstream to [0, 1] if it's the first parameter
            p0s[:,0] = np.clip(p0s[:,0], 1e-10, 1 - 1e-10)
                
            start = time.time()
            es = emcee.EnsembleSampler(
                self.meta.nwalkers, len(self.meta.flat_p0_guess), stream_funcs.spline_lnprob_1D,
                args=(self.meta.prior_arr, self.meta.phi1_spline_points, 
                    self.meta.stream.data.desi_data['VGSR'].values, self.meta.stream.data.desi_data['VRAD_ERR'].values,
                    self.meta.stream.data.desi_data['FEH'].values, self.meta.stream.data.desi_data['FEH_ERR'].values,
                    self.meta.stream.data.desi_data['PMRA'].values, self.meta.stream.data.desi_data['PMRA_ERROR'].values,
                    self.meta.stream.data.desi_data['PMDEC'].values, self.meta.stream.data.desi_data['PMDEC_ERROR'].values,
                    self.meta.stream.data.desi_data['phi1'].values, 
                    True, False, True, self.meta.spline_k, self.meta.array_lengths,
                    self.meta.vgsr_trunc, self.meta.feh_trunc, self.meta.pmra_trunc, self.meta.pmdec_trunc),
                pool=pool, backend=self.backend)
            PP = es.run_mcmc(p0s, nburnin)
            print(f'Took {(time.time()-start):.1f} seconds ({(time.time()-start)/60:.1f} minutes)')
            
            print(f'Now sampling with {nstep} iterations')
            es.reset()
            start = time.time()
            es.run_mcmc(PP.coords, nstep)
            print(f'Took {(time.time()-start):.1f} seconds ({(time.time()-start)/60:.1f} minutes)')
            
            self.chain = es.chain
            print('Getting flatchain...')
            self.flatchain = es.flatchain
    
    def show_chains(self):
        indices = np.arange(1, self.meta.no_of_spline_points + 1).astype(str)
        velocity_labels = ['vgsr' + i for i in indices]
        pmra_labels = ['pmra' + i for i in indices] 
        pmdec_labels = ['pmdec' + i for i in indices]

        self.expanded_param_labels = (['pstream'] + 
                                velocity_labels + 
                                ['lsigvgsr', 'feh1', 'lsigfeh'] +
                                pmra_labels + 
                                ['lsigpmra'] +
                                pmdec_labels +
                                ['lsigpmdec', 'bv', 'lsigbv', 'bfeh', 'lsigbfeh', 'bpmra', 'lsigbpmra', 'bpmdec', 'lsigbpmdec'])

        Nrow = self.chain.shape[2]
        fig, axes = plt.subplots(Nrow, figsize=(6,2*Nrow))


        for iparam,ax in enumerate(axes):
            for j in range(self.meta.nwalkers):
                ax.plot(self.chain[j,:,iparam], lw=.5, alpha=.2)
                ax.set_ylabel(self.expanded_param_labels[iparam])

        fig.tight_layout()

    def show_corner(self):
        flatchain = self.flatchain
        flatchain.shape
        fig = corner.corner(flatchain, labels=self.expanded_param_labels, quantiles=[0.16,0.50,0.84], show_titles=True)

    def print_result(self):
        result = stream_funcs.process_chain(self.flatchain, labels = self.expanded_param_labels)
        if len(result) == 2:
            self.meds, self.errs = result
        else:
            self.meds, self.errs, _ = result
        print(len(self.meds))
        print(self.meds)
        
        exp_flatchain = np.copy(self.flatchain)
        for i, label in enumerate(self.meds.keys()):
            if label[0] == 'l':
                exp_flatchain[:,i]= 10 ** exp_flatchain[:,i]
        result = stream_funcs.process_chain(exp_flatchain, labels = self.expanded_param_labels)
        if len(result) == 2:
            self.exp_meds, self.exp_errs = result
        else:
            self.exp_meds, self.exp_errs, _ = result
            
        result = stream_funcs.process_chain(self.flatchain, avg_error=False, labels = self.expanded_param_labels)
        if len(result) == 2:
            _, self.ep = result
            self.em = None
        else:
            _, self.ep, self.em = result
            
        exp_flatchain = np.copy(self.flatchain)
        for i, label in enumerate(self.meds.keys()):
            if label[0] == 'l':
                exp_flatchain[:,i]= 10 ** exp_flatchain[:,i]
        result = stream_funcs.process_chain(exp_flatchain, avg_error=False, labels = self.expanded_param_labels)
        if len(result) == 2:
            _, self.exp_ep = result
            self.exp_em = None
        else:
            _, self.exp_ep, self.exp_em = result

        i = 0
        # print("{:<10} {:>10} {:>10} {:>10} {:>10}".format('param','med','err','exp(med)','exp(err)'))
        print("{:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format('param','med', 'em','ep','exp(med)', 'exp(em)','exp(ep)'))
        print('--------------------------------------------------------------------------------------')
        for label,v in self.meds.items():
            # if label[:8] == 'lpstream':
            #     print("{:<10} {:>10.3f} {:>10.3f} {:>10.5f} {:>10.5f}".format(label,v,errs[label], np.e**v, np.log(10)*(np.e**v)*errs[label]))
            if label[0] == 'l':
                # print("{:<10} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} ".format(label,v,errs[label], exp_meds[label], exp_errs[label]))
                print("{:<10} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f}".format(label,v,self.em[label],self.ep[label], self.exp_meds[label], self.exp_em[label], self.exp_ep[label]))
            else:
                print("{:<10} {:>10.3f} {:>10.3f} {:>10.3f}".format(label, v, self.em[label], self.ep[label]))
            i += 1

    def memprob(self):
        #Calculate membership probabilities using the new spline_memprob_1D function
        from stream_functions import spline_memprob_1D

        # Get the data from the stream object that was optimized
        data = self.stream.data.desi_data

        # Extract the relevant parameters from the MCMC results
        theta_final = list(self.meds.values())  # Use the median parameters from MCMC

        # Calculate membership probabilities
        stream_prob = stream_funcs.spline_memprob_1D(
            theta=theta_final,
            spline_x_points=self.meta.phi1_spline_points,
            pstream_spline_x_points=self.meta.phi1_spline_points,  # Use same spline points for pstream
            lsig_vgsr_spline_points=self.meta.phi1_spline_points,  # Use same spline points for lsig_vgsr
            vgsr=data['VGSR'].values,
            vgsr_err=data['VRAD_ERR'].values,
            feh=data['FEH'].values,
            feh_err=data['FEH_ERR'].values,
            pmra=data['PMRA'].values,
            pmra_err=data['PMRA_ERROR'].values,
            pmdec=data['PMDEC'].values,
            pmdec_err=data['PMDEC_ERROR'].values,
            phi1=data['phi1'].values,
            trunc_fit=True,  # Use truncated fitting as in your setup
            reshape_arr_shape=self.meta.array_lengths,
            k=self.meta.spline_k,
            vgsr_trunc=self.meta.vgsr_trunc,
            feh_trunc=self.meta.feh_trunc,
            pmra_trunc=self.meta.pmra_trunc,
            pmdec_trunc=self.meta.pmdec_trunc
        )

        print(f"Calculated membership probabilities for {len(stream_prob)} stars")
        print(f"Membership probabilities range from {np.min(stream_prob):.3f} to {np.max(stream_prob):.3f}")
        print(f"Mean membership probability: {np.mean(stream_prob):.3f}")
        print(f"Stars with >50% probability: {len(stream_prob[stream_prob > 0.5])}")
        print(f"Stars with >70% probability: {len(stream_prob[stream_prob > 0.7])}")
        print(f"Stars with >90% probability: {len(stream_prob[stream_prob > 0.9])}")
        
        return stream_prob
    
    def save_run(self):
        """
        Save MCMC results and membership probabilities to files.
        This method saves various outputs from the MCMC run including chains,
        parameters, and high-probability stream members.
        """
        with open(self.output_dir + 'isochrone_path.txt', 'w') as f:
            f.write(self.stream.isochrone_path)

        mcmc_dict = {
            "flatchain": self.flatchain,
            "extended_param_labels": self.expanded_param_labels,
        }

        importlib.reload(stream_funcs)
        theta_final = []
        exp_theta_final = []
        errs_list = []
        exp_errs_list = []
        ep_list = []
        em_list = []
        exp_ep_list = []
        exp_em_list = []

        for label, i in self.meds.items():
            theta_final.append(i)
            exp_theta_final.append(self.exp_meds[label])
            errs_list.append(self.errs[label])
            exp_errs_list.append(self.exp_errs[label])
            ep_list.append(self.ep[label])
            em_list.append(np.abs(self.em[label]))
            exp_ep_list.append(self.exp_ep[label])
            exp_em_list.append(np.abs(self.exp_em[label]))

        nested_list_meds = stream_funcs.reshape_arr(theta_final, self.meta.array_lengths)
        nested_list_exp_meds = stream_funcs.reshape_arr(exp_theta_final, self.meta.array_lengths)
        nested_list_errs = stream_funcs.reshape_arr(errs_list, self.meta.array_lengths)
        nested_list_exp_errs = stream_funcs.reshape_arr(exp_errs_list, self.meta.array_lengths)
        nested_list_ep = stream_funcs.reshape_arr(ep_list, self.meta.array_lengths)
        nested_list_em = stream_funcs.reshape_arr(em_list, self.meta.array_lengths)
        nested_list_exp_ep = stream_funcs.reshape_arr(exp_ep_list, self.meta.array_lengths)
        nested_list_exp_em = stream_funcs.reshape_arr(exp_em_list, self.meta.array_lengths)

        nested_dict = {
            "meds": nested_list_meds,
            "exp_meds": nested_list_exp_meds,
            "errs": nested_list_errs,
            "exp_errs": nested_list_exp_errs,
            "ep": nested_list_ep,
            "em": nested_list_em,
            "exp_ep": nested_list_exp_ep,
            "exp_em": nested_list_exp_em,
            "array_lengths": self.meta.array_lengths,
            "param_labels": self.meta.param_labels,
            "expanded_param_labels": self.expanded_param_labels
        }

        np.save(f'{self.output_dir}/mcmc_dict.npy', mcmc_dict)
        np.save(f'{self.output_dir}/nested_dict.npy', nested_dict)
        np.savetxt(self.output_dir + '/' + self.stream.streamName + '_' + str(getattr(self, 'phi2_wiggle', 'default')) + '.txt', np.array(theta_final))

        # Calculate membership probabilities if not already done
        if not hasattr(self, 'stream_prob'):
            self.stream_prob = self.memprob()
        
        dataframe = self.stream.data.desi_data.copy()
        dataframe['stream_prob'] = self.stream_prob

        # Default minimum probability threshold
        min_prob = getattr(self, 'min_prob', 0.5)
        
        # Save high-probability members (above min_prob threshold)
        high_prob_mask = self.stream_prob >= min_prob
        high_prob_dataframe = dataframe[high_prob_mask]
        high_prob_table = Table.from_pandas(high_prob_dataframe)
        output_path = f'{self.output_dir}/{self.stream.streamName}_phi2_spline_{int(min_prob*100)}%_mem.fits'
        high_prob_table.write(output_path, format='fits', overwrite=True)
        print(f"Saved {len(high_prob_dataframe)} high-probability members to: {output_path}")

        # Save all stars with membership probabilities
        all_table = Table.from_pandas(dataframe)
        output_path = f'{self.output_dir}/{self.stream.streamName}_phi2_spline_all%_mem.fits'
        all_table.write(output_path, format='fits', overwrite=True)
        print(f"Saved {len(dataframe)} total stars to: {output_path}")
