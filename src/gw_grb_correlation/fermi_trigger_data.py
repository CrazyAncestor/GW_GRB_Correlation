import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib.ticker import LogFormatterMathtext
import re
from fermi_download_data_functions import download_data

def preprocess_trigger_data(year_start, year_end):
    # Create an empty DataFrame with specific column names
    detectors = [f"n{i}" for i in range(10)] + ["na", "nb", "b0", "b1"]
    columns = ['ID'] + [f"{detector}_TRIG" for detector in detectors]
    trigger_data = pd.DataFrame(columns=columns)

    for year in range(year_start, year_end):
        output_dir='trigger'
        # Download data
        download_data(range(year, year + 1), Daily_or_Burst='Burst', url_file_string="glg_bcat_all", output_dir=output_dir)

        # Process data
        trigger_data = process_fits_folder(f"./fermi_data/{output_dir}", trigger_data)

        # Delete all FITS files after processing
        fits_folder_path = f"./fermi_data/{output_dir}"
        for file in os.listdir(fits_folder_path):
            if file.endswith(('.fit', '.fits')):
                os.remove(os.path.join(fits_folder_path, file))

        print(f"Processed and saved data for {year}")
        print(trigger_data.shape)

        # Save processed data as a NumPy file
        npy_file_name = f"./fermi_data/{output_dir}/trigger_data_{year}.npy"
        np.save(npy_file_name, trigger_data.to_numpy())  # Save as .npy file

    # Save processed data as a NumPy file
    npy_file_name = f"./fermi_data/{output_dir}/trigger_data.npy"
    np.save(npy_file_name, trigger_data.to_numpy())  # Save as .npy file
    return trigger_data

def extract_fits_data(fits_file):
    """
    Extracts relevant data from the FITS file.
    
    Parameters:
    fits_file (str): Path to the FITS file.
    
    Returns:
    tuple: Returns the extracted data.
    """
    with fits.open(fits_file) as hdul:
        # Extract ID from the filename
        id = re.search(r'bn\d{9}', hdul[0].header['FILENAME'])
        id = id.group(0) if id else ""

        # Extract and check TSTART, TSTOP, T90 from header
        trigger_detectors = hdul[1].data['DETNAM']
        def extract_triggered(input_list):
        # Initialize a 14-element boolean array: 12 for NAI, 2 for BGO
            triggered = [0] * 14

            for item in input_list:
                match = re.search(r'^(NAI|BGO)_(\d+)$', item)
                if match:
                    prefix, num = match.groups()
                    idx = int(num)
                    if prefix == 'NAI' and 0 <= idx <= 11:
                        triggered[idx] = 1
                    elif prefix == 'BGO' and 0 <= idx <= 1:
                        triggered[12 + idx] = 1

            return triggered
        result = [id] + extract_triggered(trigger_detectors)

        return tuple(result)

# Function to process all FITS files in a folder and store the data in a DataFrame
def process_fits_folder(fits_folder, df=None):
    # Get all FITS files in the folder
    files = [f for f in os.listdir(fits_folder) if f.endswith(('.fit', '.fits'))]
    
    # Define columns based on data type
    detectors = [f"n{i}" for i in range(10)] + ["na", "nb", "b0", "b1"]
    columns = ['ID'] + [f"{detector}_TRIG" for detector in detectors]
    # Process files concurrently using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_fits_data, os.path.join(fits_folder, f)) for f in files]
        df = []
        for future in as_completed(futures):
            data = future.result()
            df.append(data)
        df = pd.DataFrame(df, columns=columns)
    print(df.head(5))
    return df

if __name__ == "__main__":
    preprocess_trigger_data(2015, 2026)
