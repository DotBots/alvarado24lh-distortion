import pandas as pd
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from skspatial.objects import Plane
import numpy as np

########################################################################
###                            Functions                             ###
########################################################################

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2 """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

#########################################################################
###                              Main                                 ###
#########################################################################

#########################################################################
###                        Calibration Data                           ###
#########################################################################

# Sample TSV data (you would replace this with your actual TSV file content)
tsv_file = "scene_4_3D/scene_4_3D_calib_6D.tsv"
tsv_data_file = "scene_4_3D/scene_4_3D_data_6D.tsv"
# Read the TSV data into a DataFrame
df = pd.read_csv(tsv_file, sep='\t', skiprows=13)
start_time = pd.read_csv(tsv_file, sep='\t', skiprows = lambda x: x not in [7]).columns[1]

# convert timestamp to UTC
start_time = datetime.strptime(start_time, '%Y-%m-%d, %H:%M:%S.%f')
start_time = start_time.replace(tzinfo=ZoneInfo("Europe/Paris"))
start_time = start_time.astimezone(ZoneInfo("UTC"))

# Select only the first 5 columns
df = df.iloc[:, :5]
#CHange the names of the columns and reorder the columns
df.rename(columns={'DotBox X': 'x', 'Y':'y', 'Z':'z', 'Time':'timestamp','Frame':'frame'}, inplace=True)
df = df[['timestamp','frame','x','y','z']]
# Drop rows with 0.0 readings
df = df[(df['x'] != 0.0) | (df['y'] != 0.0) | (df['z'] != 0.0)]
df.reset_index(drop=True, inplace=True)


# Convert timetamp column to a datetime object
df['timestamp'] = df['timestamp'].apply(lambda x: start_time + timedelta(seconds=x))
df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

# Save the selected columns to a CSV file
csv_file_path = 'scene_4_3D_calib.csv'  # Specify your desired file path here
df.to_csv(csv_file_path, index=True)

print(f"File saved to {csv_file_path}")

#########################################################################
###                             Data File                             ###
#########################################################################

# Sample TSV data (you would replace this with your actual TSV file content)
tsv_data_file = "scene_4_3D/scene_4_3D_data_6D.tsv"
# Read the TSV data into a DataFrame
df_data = pd.read_csv(tsv_data_file, sep='\t', skiprows=13)
data_start_time = pd.read_csv(tsv_data_file, sep='\t', skiprows = lambda x: x not in [7]).columns[1]

# convert timestamp to UTC
data_start_time = datetime.strptime(data_start_time, '%Y-%m-%d, %H:%M:%S.%f')
data_start_time = data_start_time.replace(tzinfo=ZoneInfo("Europe/Paris"))
data_start_time = data_start_time.astimezone(ZoneInfo("UTC"))

# Select only the first 5 columns
df_data = df_data.iloc[:, :5]
#CHange the names of the columns and reorder the columns
df_data.rename(columns={'DotBox X': 'x', 'Y':'y', 'Z':'z', 'Time':'timestamp','Frame':'frame'}, inplace=True)
df_data = df_data[['timestamp','frame','x','y','z']]
# Drop rows with 0.0 readings
df_data = df_data[(df_data['x'] != 0.0) | (df_data['y'] != 0.0) | (df_data['z'] != 0.0)]
df_data.reset_index(drop=True, inplace=True)

# Convert timetamp column to a datetime object
df_data['timestamp'] = df_data['timestamp'].apply(lambda x: data_start_time + timedelta(seconds=x))

# Clear outliers


# Convert timetamp column from a datetime object to a properly formated string
df_data['timestamp'] = df_data['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

# Concatenate the calib and data files to export a single big file
df_data = pd.concat([df, df_data], ignore_index=True)




# Save the selected columns to a CSV file
csv_data_file_path = 'scene_4_3D_data.csv'  # Specify your desired file path here
df_data.to_csv(csv_data_file_path, index=True)

print(f"File saved to {csv_data_file_path}")