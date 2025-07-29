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
import os
import feh_correct
import warnings
from astropy.utils.exceptions import AstropyDeprecationWarning

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
        'TARGET_RA', 'TARGET_DEC', 'FEH', 'FEH_ERR', 
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

        # Lets load the STREAMFINDER data for Gaia DR3
        if sf_path:
            sf_data = table.Table.read(self.sf_path)
            self.sf_data = sf_data.to_pandas()
        else: 
            print('No STREAMFINDER path given.')

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
            base_name = 'confirmed_sf_and_desi'
            attr_name = base_name
            if hasattr(self, base_name):
                import string
                suffix = 'b'
                while hasattr(self, f'{base_name}_{suffix}'):
                    suffix = chr(ord(suffix) + 1)
                attr_name = f'{base_name}_{suffix}'

            # Perform the merge and assign to the chosen attribute
            merged = pd.merge(
                self.SoI_streamfinder.drop_duplicates(subset=[gaia_source_ids]),
                self.desi_data.drop_duplicates(subset=['SOURCE_ID']),
                left_on=gaia_source_ids,
                right_on='SOURCE_ID',
                how='inner'
            )
            merged.dropna(inplace=True)

            setattr(self, attr_name, merged)
            
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
            how='left',
            indicator=True
            )

            # Keep only the SoI_streamfinder rows that do not match any in desi_data
            only_in_SoI = unmatched[unmatched['_merge'] == 'left_only'].drop(columns=['SOURCE_ID', '_merge'])
            setattr(self, attr_name, only_in_SoI)

class stream:
    def __init__(self, data_object, streamName='Sylgr-I21', streamNo=42):
        self.streamName = streamName
        self.streamNo = streamNo

        # Store a reference to the data object instead of re-running the init
        self.data = data_object

        # Now, access the dataframes through the passed object
        # e.g., self.data.sf_data instead of self.sf_data
        self.data.SoI_streamfinder = self.data.sf_data[self.data.sf_data['Stream'] == self.streamNo]
        self.data.SoI_streamfinder.label='SF3'

        print('Importing galstreams module...')
        import galstreams
        mwsts = galstreams.MWStreams(verbose=False, implement_Off=True)
        self.data.SoI_galstream = mwsts.get(streamName, None)
        self.data.SoI_galstream.label = 'galstream'

        print('Creating combined DataFrame of SF and DESI')
        # Access desi_data through self.data
        self.data.sfCrossMatch() #saved as confirmed_sf_and_desi
        self.data.sfCrossMatch(isin=False) #creates DF of stars not in DESI

        self.frame = self.data.SoI_galstream.stream_frame
        self.data.SoI_galstream.gal_phi1 = self.data.SoI_galstream.track.transform_to(self.frame).phi1.deg
        self.data.SoI_galstream.gal_phi2 = self.data.SoI_galstream.track.transform_to(self.frame).phi2.deg
            

        self.data.desi_data['phi1'], self.data.desi_data['phi2'] = stream_funcs.ra_dec_to_phi1_phi2(self.frame, np.array(self.data.desi_data['TARGET_RA'])*u.deg, np.array(self.data.desi_data['TARGET_DEC'])*u.deg)

        self.data.confirmed_sf_and_desi['phi1'], self.data.confirmed_sf_and_desi['phi2'] = stream_funcs.ra_dec_to_phi1_phi2(self.frame,np.array(self.data.confirmed_sf_and_desi['TARGET_RA'])*u.deg, np.array(self.data.confirmed_sf_and_desi['TARGET_DEC'])*u.deg)


#class Orbit: WIP


class StreamPlotter:
    """
    For really clean and easy plotting
    """
    def __init__(self, stream_object, save_dir='plots/'):
        """
        Initializes the plotter with a stream object.
        
        Args:
            stream_object (stream): An instance of your stream class.
            save_dir (str): Directory to save plots.
        """
        self.stream = stream_object
        self.data = stream_object.data
        self.save_dir = save_dir
        # Create directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)


        self.plot_params = {
            'sf_in_desi': {
                'marker': 'd',
                's': 30,
                'color': 'green',
                'label': r'SF $\in$ DESI',
                'edgecolor': 'k',
                'zorder': 5 # zorder ensures these points are plotted on top
            },
            'sf_not_desi': {
                'marker': 'd',
                's': 15,
                'color': 'orange',
                'alpha': 0.8,
                'label': 'SF (not in DESI)',
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
                'color': 'y', # You can still reference old style if you wish
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
            }
        }
    
    def on_sky(self, stream_frame=True, save=False, galstream=True, orbit=True):
        """
        Plots the stream on-sky either in RA, DEC or phi1, phi2
        """
        if stream_frame:
            col_x = 'phi1'
            label_x = r'$\phi_1$'
            col_y = 'phi2'
            label_y = r'$\phi_2$'
        else:
            col_x = 'TARGET_RA'
            col_y = 'TARGET_DEC'
            label_x = 'RA (deg)'
            label_y = 'DEC (deg)'

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.scatter(
            self.data.confirmed_sf_and_desi[col_x],
            self.data.confirmed_sf_and_desi[col_y],
            **self.plot_params['sf_in_desi'] # <-- Unpacking the dictionary
        )
        # ax.scatter(
        #     self.data.confirmed_sf_not_desi[col_x],
        #     self.data.confirmed_sf_not_desi[col_y],
        #     **self.plot_params['sf_not_desi'] # <-- Unpacking the dictionary
        # )
        if stream_frame:
            if galstream:
                ax.plot(
                    self.data.SoI_galstream.gal_phi1,
                    self.data.SoI_galstream.gal_phi2,
                    **self.plot_params['galstream_track'])
        else:
            if galstream:
                ax.plot(
                    self.data.SoI_galstream.track.ra,
                    self.data.SoI_galstream.track.dec,
                    **self.plot_params['galstream_track'])
        
        # if hasattr(self.stream, orbit): WIP
        #         ax.plot(
        #             self.orbit.,
        #             self.orbit,
        #             **self.plot_params['orbit_track'])

        
        ax.legend()
        ax.set_ylabel(label_y)
        ax.set_xlabel(label_x)
        stream_funcs.plot_form(ax)



        


