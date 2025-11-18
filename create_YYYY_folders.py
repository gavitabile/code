# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:43:47 2025

@author: GAVITAB
"""

# 1. Adjust tickers as needed (remove or add entities) (subfolder_names = [ ])
# 2. Adjust YY (months = [f"26...])
# 3. Adjust YYYY (year_folder = )
# The code is set for 2026 already, just run!

import os

# Base path
base_path = r"I:\1. ALM\Analysen"

# Subfolder names
subfolder_names = [
    "AB", "AD", "AE", "AF", "AG", "AI", "AL", "AN", "AO", "AP", "AT", "AU", "BE", "CA", "CH", "CN", "CZ", "DK", "ES", "FL", "FR", "GB", "HK", "HU", "IN", "IT", "JP", "KM", "KR", "LD", "MBAM", "MC", "MY", "NL", "NZ", "PL", "PT", "RO", "SE", "SG", "SK", "TH", "TR", "TW", "US", "WP", "ZA"
    # Add more subfolder names as needed
]

# Generate the folder paths dynamically
folders = [fr"{base_path}\{name}" for name in subfolder_names]

# Names for subfolders
months = [f"26{str(i).zfill(2)}" for i in range(1, 13)]  # Generates 2601 to 2612
subfolders = ["BSTB", "Mail", "Fiti", "Nachsimulation"]  # Subfolder names

# Create the directory structure
for base_folder in folders:
    year_folder = os.path.join(base_folder, "2026")
    try:
        os.makedirs(year_folder, exist_ok=True)  # Create '2025' folder
        for month in months:
            month_folder = os.path.join(year_folder, month)
            os.makedirs(month_folder, exist_ok=True)  # Create month folder
            for subfolder in subfolders:
                subfolder_path = os.path.join(month_folder, subfolder)
                os.makedirs(subfolder_path, exist_ok=True)  # Create subfolders
        print(f"Successfully created structure in: {base_folder}")
    except Exception as e:
        print(f"Failed to create structure in {base_folder}: {e}")