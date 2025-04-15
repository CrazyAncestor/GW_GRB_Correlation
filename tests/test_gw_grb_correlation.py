#!/usr/bin/env python

# Add src/ to sys.path
script_dir = os.path.dirname(__file__)
src_path = os.path.abspath(os.path.join(script_dir, '..', 'gw_grb_correlation'))
sys.path.insert(0, src_path)

from fermi_data_preprocessing import download_and_preprocess_fermi_data

start_year = 2015
end_year = 2026
fermi_data = download_and_preprocess_fermi_data(start_year=start_year, end_year=end_year, download_or_not=False)