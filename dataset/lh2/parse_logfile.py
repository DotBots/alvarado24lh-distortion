import pandas as pd
from datetime import datetime
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2

#############################################################################
###                                Options                                ###
#############################################################################

scene_number = 3

#############################################################################
###                                Code                                   ###
#############################################################################

filename_calib = f"./scene_{scene_number}/scene_{scene_number}_calib.log"
filename_data  = f"./scene_{scene_number}/scene_{scene_number}_data.log"

if scene_number == 4: 
    filename_calib = f"./scene_{scene_number}_3D/scene_{scene_number}_3D_calib.log"
    filename_data  = f"./scene_{scene_number}_3D/scene_{scene_number}_3D_data.log"

## Read the struct log with the information
# Define a regular expression pattern to extract timestamp and source from log lines
log_pattern = re.compile(r'timestamp=(?P<timestamp>.*?) .*? source=(?P<source>\S+) .*? sweep_0_poly=(?P<sweep_0_poly>\d+) sweep_0_off=(?P<sweep_0_off>\d+) sweep_0_bits=(?P<sweep_0_bits>\d+) sweep_1_poly=(?P<sweep_1_poly>\d+) sweep_1_off=(?P<sweep_1_off>\d+) sweep_1_bits=(?P<sweep_1_bits>\d+) sweep_2_poly=(?P<sweep_2_poly>\d+) sweep_2_off=(?P<sweep_2_off>\d+) sweep_2_bits=(?P<sweep_2_bits>\d+) sweep_3_poly=(?P<sweep_3_poly>\d+) sweep_3_off=(?P<sweep_3_off>\d+) sweep_3_bits=(?P<sweep_3_bits>\d+)')

# Create an empty list to store the extracted data
data = []

# Open the log file and iterate over each line
for filename in [filename_calib, filename_data]:
    with open(filename, "r") as log_file:
        for line in log_file:
            # Extract timestamp and source from the line
            match = log_pattern.search(line)
            if match and "lh2-4" in line:
                # Append the extracted data to the list
                data.append({
                    "timestamp": datetime.strptime(match.group("timestamp"), "%Y-%m-%dT%H:%M:%S.%fZ"),
                    "source": match.group("source"),
                    "poly_0": int(match.group("sweep_0_poly")),
                    "off_0":  int(match.group("sweep_0_off")),
                    "bits_0": int(match.group("sweep_0_bits")),
                    "poly_1": int(match.group("sweep_1_poly")),
                    "off_1":  int(match.group("sweep_1_off")),
                    "bits_1": int(match.group("sweep_1_bits")),
                    "poly_2": int(match.group("sweep_2_poly")),
                    "off_2":  int(match.group("sweep_2_off")),
                    "bits_2": int(match.group("sweep_2_bits")),
                    "poly_3": int(match.group("sweep_3_poly")),
                    "off_3":  int(match.group("sweep_3_off")),
                    "bits_3": int(match.group("sweep_3_bits")),
                })

# Create a pandas DataFrame from the extracted data
df = pd.DataFrame(data)

## Remove lines that don't have the data from both lighthouses
# Define the conditions
cond1 = df[['poly_0', 'poly_1', 'poly_2', 'poly_3']].isin([0, 1]).sum(axis=1) == 2
cond2 = df[['poly_0', 'poly_1', 'poly_2', 'poly_3']].isin([2, 3]).sum(axis=1) == 2
cond = cond1 & cond2
# Filter the rows that meet the condition
df = df.loc[cond].reset_index(drop=True)

## Convert the data to a numpy a array and sort them to make them compatible with Cristobal's code
poly_array = df[["bits_0", "bits_1", "bits_2", "bits_3"]].to_numpy()
sorted_indices = np.argsort(df[['poly_0','poly_1','poly_2','poly_3']].values,axis=1)
bits_df = df[['bits_0','bits_1','bits_2','bits_3']]
sorted_bits = np.empty_like(bits_df)
for i, row in enumerate(sorted_indices):
    sorted_bits[i] = bits_df.values[i, row]


## Sort the columns for LH2-A and LH2-B separatedly.
c01 = np.sort(sorted_bits[:,0:2], axis=1).astype(int)
c23 = np.sort(sorted_bits[:,2:4], axis=1).astype(int)
# Re-join the columns and separate them into the variables used by cristobals code.
c0123 = np.hstack([c01, c23])
c0123 = np.sort(sorted_bits, axis=1).astype(int)
# This weird order to asign the columns is because there was an issue with the dataset, and the data order got jumbled.
c1A = c0123[:,0] 
c2A = c0123[:,2]
c1B = c0123[:,1]
c2B = c0123[:,3]


#############################################################################
###                           Save reordered data                         ###
#############################################################################

sorted_df = pd.DataFrame({
                          'timestamp' : df['timestamp'],

                          'LHA_count_1': c0123[:,0],

                          'LHA_count_2': c0123[:,2],

                          'LHB_count_1': c0123[:,1],

                          'LHB_count_2': c0123[:,3]},
                          index = df.index
                          )

#############################################################################
###                           Clear Outliers                         ###
#############################################################################
# This goes grid point by grid point and removes datapoints who are too far away from mean.

def clear_outliers(df, threshold=5e3):
    """
    takes a dataframe with the following coulmns 
    "timestamp", 'LHA_count_1', 'LHA_count_2', 'LHB_count_1', 'LHB_count_2'
    and removes any rows in which a change of more than 10k units per second occur.
    """


    # Function to calculate the rate of change
    def rate_of_change(row, prev_row):
        time_diff = (row['timestamp'] - prev_row['timestamp']).total_seconds()
        if time_diff > 0:
            for col in ['LHA_count_1', 'LHA_count_2', 'LHB_count_1', 'LHB_count_2']:
                rate = abs(row[col] - prev_row[col]) / time_diff
                if rate > threshold:
                    return True
        return False
    
    def check_jump(row, prev_row, next_row):
        for col in ['LHA_count_1', 'LHA_count_2', 'LHB_count_1', 'LHB_count_2']:
            if abs(row[col] - prev_row[col]) > threshold and abs(next_row[col] - row[col]) > threshold:
                return True
        return False

    should_restart = True
    while should_restart:
        should_restart = False
        index_list = df.index.tolist()
        for i in range(len(index_list)):

            if i == 0:
                continue
            # for i, row in df.iterrows():

            # Check for quikly changing outputs
            if rate_of_change(df.loc[index_list[i]], df.loc[index_list[i-1]]):
                df.drop(index_list[i], axis=0, inplace=True)
                should_restart = True
                break

            # Check for individual peaks, 1-row deltas (don't run on the last index)
            if i != len(index_list)-1:
                if check_jump(df.loc[index_list[i]], df.loc[index_list[i-1]], df.loc[index_list[i+1]]):
                    df.drop(index_list[i], axis=0, inplace=True)
                    should_restart = True
                    break

            # Check for any row with a 0
            if df.loc[index_list[i]].eq(0).any():
                df.drop(index_list[i], axis=0, inplace=True)
                should_restart = True
                break

    return df


# Get the cleaned values back on the variables needed for the next part of the code.
sorted_df = clear_outliers(sorted_df, 10e3)


# Change the format of the timestamp column
sorted_df['timestamp'] = sorted_df['timestamp'].apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))


# Save the dataframe to a csv
if scene_number == 4:
    sorted_df.to_csv(f'./LH_data_scene_{scene_number}_3D.csv', index=True)
else:
    sorted_df.to_csv(f'./LH_data_scene_{scene_number}.csv', index=True)


a=0
