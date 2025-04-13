import os
import numpy as np
import pandas as pd


from fermi_time_data import preprocess_time_data
from fermi_location_data import preprocess_location_data
from fermi_tte_data import preprocess_tte_data
from fermi_poshist_data import preprocess_poshist_data, interpolate_qs_for_time

def create_dataframe_and_name_column_from_data_files(data_type, PRINT_HEAD = False):
    """
    Loads a .npy file into a Pandas DataFrame with allow_pickle=True and prints its info and head.
    :param data_type: Identifier for the dataset (e.g., 'time', 'tte', 'location')
    :return: Loaded DataFrame
    """
    
    file_path = f'./fermi_data/{data_type}/{data_type}_data.npy'
    df = pd.DataFrame(np.load(file_path, allow_pickle=True))
    if data_type=='time':
        df.columns = ['ID', 'TSTART', 'TSTOP', 'T90', 'DATE']
    elif data_type =='tte':
        detectors = [f"n{i}" for i in range(10)] + ["na", "nb", "b0", "b1"]
        df.columns = ['ID'] + [f"{detector}_PH_CNT" for detector in detectors]
    elif data_type == 'location':
        df.columns = ['ID', 'RA', 'DEC']
    elif data_type =='poshist':
        df.columns = ['TSTART', 'QSJ_1', 'QSJ_2','QSJ_3','QSJ_4']
    elif data_type =='fermi':
        detectors = [f"n{i}" for i in range(10)] + ["na", "nb", "b0", "b1"]
        df.columns = ['ID', 'TSTART', 'TSTOP', 'T90', 'DATE']  + [f"{detector}_PH_CNT" for detector in detectors] + ['RA', 'DEC'] + ['QSJ_1', 'QSJ_2','QSJ_3','QSJ_4']
    if PRINT_HEAD:
        print(f"\nData from {file_path}:")
        print(df.info())
        print(df.head())
    return df

def merge_all_datatypes_in_fermi(time_df, tte_df, location_df, poshist_df, print_info = False):
    """
    Merges time, tte, and location DataFrames on a common ID using an inner join.
    :param time_df: DataFrame containing time data
    :param tte_df: DataFrame containing tte data
    :param location_df: DataFrame containing location data
    :return: Merged DataFrame
    """
    
    merged_df = time_df.merge(tte_df, on='ID', how='inner')
    merged_df = merged_df.merge(location_df, on='ID', how='inner')
    GRB_poshist = interpolate_qs_for_time(poshist_df.astype(float), merged_df['TSTART'].astype(float))
    
    merged_df = merged_df.merge(GRB_poshist, on='TSTART', how='inner')

    if print_info:
        print("\nMerged Data:")
        print(merged_df.info())
        print(merged_df.head())

    return merged_df

def download_and_preprocess_fermi_data(start_year, end_year, download_or_not = True):
    
    if download_or_not:
        # Download and preprocess raw data
        time_data = preprocess_time_data(start_year, end_year)
        location_data = preprocess_location_data(start_year, end_year)
        tte_data = preprocess_tte_data(start_year, end_year)
        poshist_data = preprocess_poshist_data(start_year, end_year)

    # Load and display the data
    time_data = create_dataframe_and_name_column_from_data_files('time')
    location_data = create_dataframe_and_name_column_from_data_files('location')
    tte_data = create_dataframe_and_name_column_from_data_files('tte')
    poshist_data = create_dataframe_and_name_column_from_data_files('poshist')

    # Merge the data
    merged_data = merge_all_datatypes_in_fermi(time_data, tte_data, location_data, poshist_data)
    
    output_dir = f"./fermi_data/fermi/"
    os.makedirs(output_dir, exist_ok=True)
    # Save the merged data to a .npy file
    np.save(output_dir + "fermi_data.npy", merged_data.to_records(index=False))  # Convert to NumPy structured array
    print(f"\nPreprocessed data saved to {output_dir}")
    
    # Save the merged data to a .csv file
    merged_data.to_csv(output_dir + "fermi_data.csv", index=False)  # Save without row indices
    print(f"\nPreprocessed data saved to {output_dir}")

    return merged_data
    


if __name__ == "__main__":
    start_year = 2025
    end_year = 2026
    fermi_data = download_and_preprocess_fermi_data(start_year=start_year, end_year=end_year, download_or_not=True)
