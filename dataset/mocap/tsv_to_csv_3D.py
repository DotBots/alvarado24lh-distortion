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
###                            Options                                ###
#########################################################################

tsv_file = "LHB-DotBox/Measurement2_6D.tsv"
# tsv_file = "LHC-DotBox/Measurement10_6D.tsv"

#########################################################################
###                        Calibration Data                           ###
#########################################################################

# Read the TSV data into a DataFrame
df = pd.read_csv(tsv_file, sep='\t', skiprows=13)
start_time = pd.read_csv(tsv_file, sep='\t', skiprows = lambda x: x not in [7]).columns[1]

# convert timestamp to UTC
start_time = datetime.strptime(start_time, '%Y-%m-%d, %H:%M:%S.%f')
start_time = start_time.replace(tzinfo=ZoneInfo("Europe/Paris"))
start_time = start_time.astimezone(ZoneInfo("UTC"))

# Change the names of the columns and reorder the columns
df.rename(columns={'DotBox X': 'dotbot_x_mm', 'Y':'dotbot_y_mm', 'Z':'dotbot_z_mm', 'Time':'timestamp', 'Frame':'frame', \
                    'LHC X':'lh_x_mm', 'Y.2':'lh_y_mm', 'Z.2':'lh_z_mm', 'Roll.2':'lh_roll_deg',  'Pitch.2':'lh_pitch_deg', 'Yaw.2':'lh_yaw_deg',}, inplace=True)
# Keep only the important columns
df = df[['timestamp','frame','dotbot_x_mm','dotbot_y_mm','dotbot_z_mm', 'lh_x_mm', 'lh_y_mm', 'lh_z_mm', 'lh_roll_deg', 'lh_pitch_deg', 'lh_yaw_deg']]
# Drop rows with 0.0 readings
df = df[(df != 0.0).all(axis=1)]
df.reset_index(drop=True, inplace=True)


# Convert timetamp column to a datetime object
df['timestamp'] = df['timestamp'].apply(lambda x: start_time + timedelta(seconds=x))
df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

# Save the selected columns to a CSV file
csv_file_path = tsv_file.split('/')[0] + '/mocap_data.csv' # Specify your desired file path here
df.to_csv(csv_file_path, index=True)

print(f"File saved to {csv_file_path}")

# Clear outliers
