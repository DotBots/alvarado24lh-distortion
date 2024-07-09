import json
import numpy as np
import pandas as pd
from dateutil import parser
import datetime
import cv2
from skspatial.objects import Plane, Points

#############################################################################
###                                Options                                ###
#############################################################################
# If False, asuumes that the LH camera matrix is the identity.
# If True, uses Mat_A and Mat_B
USE_CAMERA_MATRIX = False    

# Define the intrinsic camera matrices of the LH2 basestations
Mat_A = np.array([[ 1.     ,  0.    , 0.],
                  [ 0.     ,  1.    , 0.],
                  [ 0.     ,  0.    , 1.]])

Mat_B = np.array([[ 1.      , 0.    , 0.],
                  [ 0.      , 0.    , 0.],
                  [ 0.      , 0.    , 1.]])

#############################################################################
###                                Functions                              ###
#############################################################################

def read_dataset_files(lh_file, mocap_file):
    """"
    Reads the files with the experiment data, one for the LH data and one for the Mocap data.
    Interpolates the Mocap data to match the LH data and combines both dataset into a single
    Pandas dataframe with the following columns:
    timestamp,LHA_count_1,LHA_count_2,LHB_count_1,LHB_count_2,real_x_mm,real_y_mm,real_z_mm
    """
    # Open all files
    lh_data = pd.read_csv(lh_file, index_col=0, parse_dates=['timestamp'])
    mocap_data = pd.read_csv(mocap_file, index_col=0, parse_dates=['timestamp'])

    # Fix the timestamp for the LH data
    base_timestamp = lh_data.iloc[0]['timestamp']
    base_db_time   = lh_data.iloc[0]['db_time']

    prev_db_time   = lh_data.iloc[0]['db_time']



    for index, row in lh_data.iterrows():
        current_timestamp = lh_data.at[index, 'timestamp']
        current_db_time   = lh_data.at[index, 'db_time']

        # A lh-minimote reset was detected. Reset the base for the timestamp calculation.
        if (current_db_time < prev_db_time):
            base_db_time = current_db_time
            base_timestamp = current_timestamp
            continue

        # Estimate the real timestamp from the initial timestamp + the dotbot timer 
        lh_data.at[index, 'timestamp'] = base_timestamp + datetime.timedelta(microseconds=float(current_db_time - base_db_time))

        # Update the previous value
        prev_db_time  = current_db_time

    a=5

   




   
    # ## Handle the calibration file
    # lh2_calib_time = calib_data["scene_4_3D"]["timestamps_lh2"]
    # # Convert the strings to datetime objects
    # for key in lh2_calib_time:
    #     lh2_calib_time[key] = [parser.parse(ts) for ts in lh2_calib_time[key]]
    # # Slice the calibration data and add it to the  data dataframe.
    # tl = lh_data.loc[ (lh_data['timestamp'] > lh2_calib_time["tl"][0]) & (lh_data['timestamp'] < lh2_calib_time["tl"][1])].mean(axis=0, numeric_only=True)
    # tr = lh_data.loc[ (lh_data['timestamp'] > lh2_calib_time["tr"][0]) & (lh_data['timestamp'] < lh2_calib_time["tr"][1])].mean(axis=0, numeric_only=True)
    # bl = lh_data.loc[ (lh_data['timestamp'] > lh2_calib_time["bl"][0]) & (lh_data['timestamp'] < lh2_calib_time["bl"][1])].mean(axis=0, numeric_only=True)
    # br = lh_data.loc[ (lh_data['timestamp'] > lh2_calib_time["br"][0]) & (lh_data['timestamp'] < lh2_calib_time["br"][1])].mean(axis=0, numeric_only=True)
    # # Save the calibration data.
    # calib_data["scene_4_3D"]['corners_lh2_count'] = {'tl':tl,
    #                              'tr':tr,
    #                              'bl':bl,
    #                              'br':br,
    #                              }


    # # Get a unix timestamp column out of the datetime.
    # for df in [lh_data, mocap_data]:
    #     df['time_s'] = df['timestamp'].apply(lambda x: x.timestamp() )

    # # slice the datasets to be in the same timeframe.
    # # Slice LH to Mocap
    # start = mocap_data['timestamp'].iloc[0]  
    # end   = mocap_data['timestamp'].iloc[-1]
    # lh_data = lh_data.loc[ (lh_data['timestamp'] > start) & (lh_data['timestamp'] < end)]
    # # Slice Mocap to LH
    # start = lh_data['timestamp'].iloc[0]  
    # end   = lh_data['timestamp'].iloc[-1]
    # mocap_data = mocap_data.loc[ (mocap_data['timestamp'] > start) & (mocap_data['timestamp'] < end)]


    # ## Interpolate the Mocap data to match the LH data.
    # mocap_np = {'time': mocap_data['time_s'].to_numpy(),
    #             'x':    mocap_data['x'].to_numpy(),
    #             'y':    mocap_data['y'].to_numpy(),
    #             'z':    mocap_data['z'].to_numpy()}
    
    # lh_time = lh_data['time_s'].to_numpy()

    # # Offset the camera timestamp to get rid of the communication delay.
    # mocap_np['time'] += 265000e-6 # seconds
    # mocap_np['x_interp_lh'] = np.interp(lh_time, mocap_np['time'],  mocap_np['x'])
    # mocap_np['y_interp_lh'] = np.interp(lh_time, mocap_np['time'],  mocap_np['y'])
    # mocap_np['z_interp_lh'] = np.interp(lh_time, mocap_np['time'],  mocap_np['z'])


    # merged_data = pd.DataFrame({
    #                       'timestamp' : lh_data['timestamp'],
    #                       'time_s' : lh_data['time_s'],
    #                       'LHA_count_1' : lh_data['LHA_count_1'],
    #                       'LHA_count_2' : lh_data['LHA_count_2'],
    #                       'LHB_count_1' : lh_data['LHB_count_1'],
    #                       'LHB_count_2' : lh_data['LHB_count_2'],
    #                       'real_x_mm': mocap_np['x_interp_lh'],
    #                       'real_y_mm': mocap_np['y_interp_lh'],
    #                       'real_z_mm': mocap_np['z_interp_lh']}
    #                       )
    
    # return merged_data, calib_data["scene_4_3D"]

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
    max_i = 0 # keeps track of the max processed index, it helps me keep track of how much time processing is taking 
    while should_restart:
        should_restart = False
        index_list = df.index.tolist()
        for i in range(max(0,max_i), len(index_list)):

            # check which is the index processed
            if i > max_i:  
                max_i = i
                # if max_i % 20 == 0: print(f"outlier detection: max_i={max_i}")

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
            if df.loc[index_list[i], ['LHA_count_1', 'LHA_count_2', 'LHB_count_1', 'LHB_count_2']].eq(0).any():
                df.drop(index_list[i], axis=0, inplace=True)
                should_restart = True
                break

    return df

def inject_noise_on_data(df, angle_std=10):

    # calculate noise std deviation in polynomial counts
    count_std = angle_std * 959000 / 180

    # Calculate normally distributed random noise in the size of the dataframe
    noise = np.random.normal(0, count_std, df[['LHA_count_1','LHA_count_2', 'LHB_count_1', 'LHB_count_2']].shape)

    # Add noise to the data frame
    df_copy = df.copy(deep=True)

    df_copy[['LHA_count_1','LHA_count_2', 'LHB_count_1', 'LHB_count_2']] += noise

    return df_copy

def LH2_count_to_pixels(count_1, count_2, mode):
    """
    Convert the sweep count from a single lighthouse into pixel projected onto the LH2 image plane
    ---
    count_1 - int - polinomial count of the first sweep of the lighthouse
    count_2 - int - polinomial count of the second sweep of the lighthouse
    mode - int [0,1] - mode of the LH2, let's you know which polynomials are used for the LSFR. and at which speed the LH2 is rotating.
    """
    periods = [959000, 957000]

    # Translate points into position from each camera
    a1 = (count_1*8/periods[mode])*2*np.pi  # Convert counts to angles traveled in the weird 40deg planes, in radians
    a2 = (count_2*8/periods[mode])*2*np.pi   

    # Transfor sweep angles to azimuth and elevation coordinates
    azimuth   = (a1+a2)/2 
    elevation = np.pi/2 - np.arctan2(np.sin(a2/2-a1/2-60*np.pi/180),np.tan(np.pi/6)) 

    # Project the angles into the z=1 image plane
    pts_lighthouse = np.zeros((len(count_1),2))
    for i in range(len(count_1)):
        pts_lighthouse[i,0] = -np.tan(azimuth[i])
        pts_lighthouse[i,1] = -np.sin(a2[i]/2-a1[i]/2-60*np.pi/180)/np.tan(np.pi/6) * 1/np.cos(azimuth[i])

    # Return the projected points
    return pts_lighthouse

def LH2_angles_to_pixels(azimuth, elevation):
    """
    Project the Azimuth and Elevation angles of a LH2 basestation into the unit image plane.
    """
    pts_lighthouse = np.array([np.tan(azimuth),         # horizontal pixel  
                               np.tan(elevation) * 1/np.cos(azimuth)]).T    # vertical   pixel 
    return pts_lighthouse

def solve_3d_scene(pts_a, pts_b):
    """
    Use the projected LH2-camera points to triangulate the position of the LH2  basestations and of the LH2 receiver
    """
    # Obtain translation and rotation vectors
    F, mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_LMEDS)
    if USE_CAMERA_MATRIX:
        points, R_star, t_star, mask = cv2.recoverPose(Mat_A @ F @ Mat_B, pts_a, pts_b)
    else:
        points, R_star, t_star, mask = cv2.recoverPose(F, pts_a, pts_b)

    # Triangulate the points
    R_1 = np.eye(3,dtype='float64')
    t_1 = np.zeros((3,1),dtype='float64')

    # Calculate the Projection Matrices P
    # The weird transpose everywhere is  because cv2.recover pose gives you the Camera 2 to Camera 1 transformation (which is the backwards of what you want.)
    # To get the Cam1 -> Cam2 transformation, we need to invert this.
    # R^-1 => R.T  (because rotation matrices are orthogonal)
    # inv(t) => -t 
    # That's where all the transpositions and negatives come from.
    # Source: https://stackoverflow.com/a/45722936
    if USE_CAMERA_MATRIX:
        P1 = Mat_A @ np.hstack([R_1.T, -R_1.T.dot(t_1)])
        P2 = Mat_B @ np.hstack([R_star.T, -R_star.T.dot(t_star)])  
    else:
        P1 = np.hstack([R_1.T, -R_1.T.dot(t_1)])
        P2 = np.hstack([R_star.T, -R_star.T.dot(t_star)])  
    # The projection matrix is the intrisic matrix of the camera multiplied by the extrinsic matrix.
    # When the intrinsic matrix is the identity.
    # The results is the [ Rotation | translation ] in a 3x4 matrix

    point3D = cv2.triangulatePoints(P1, P2, pts_b.T, pts_a.T).T
    point3D = point3D[:, :3] / point3D[:, 3:4]

    # Return the triangulated 3D points
    # Return the position and orientation of the LH2-B wrt LH2-A
    return point3D, t_star, R_star

def solve_3d_scene_get_Rt(pts_a, pts_b):
    """
    Use the projected LH2-camera points to triangulate the position of the LH2  basestations and of the LH2 receiver
    """
    # Obtain translation and rotation vectors
    F, mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_LMEDS)
    if USE_CAMERA_MATRIX:
        points, R_star, t_star, mask = cv2.recoverPose(Mat_A @ F @ Mat_B, pts_a, pts_b)
    else:
        points, R_star, t_star, mask = cv2.recoverPose(F, pts_a, pts_b)

    return t_star, R_star

def solve_3d_scene_triangulate_points(pts_a, pts_b, t_star, R_star):

   # Triangulate the points
    R_1 = np.eye(3,dtype='float64')
    t_1 = np.zeros((3,1),dtype='float64')

    # Calculate the Projection Matrices P
    # The weird transpose everywhere is  because cv2.recover pose gives you the Camera 2 to Camera 1 transformation (which is the backwards of what you want.)
    # To get the Cam1 -> Cam2 transformation, we need to invert this.
    # R^-1 => R.T  (because rotation matrices are orthogonal)
    # inv(t) => -t 
    # That's where all the transpositions and negatives come from.
    # Source: https://stackoverflow.com/a/45722936
    if USE_CAMERA_MATRIX:
        P1 = Mat_A @ np.hstack([R_1.T, -R_1.T.dot(t_1)])
        P2 = Mat_B @ np.hstack([R_star.T, -R_star.T.dot(t_star)])  
    else:
        P1 = np.hstack([R_1.T, -R_1.T.dot(t_1)])
        P2 = np.hstack([R_star.T, -R_star.T.dot(t_star)])  
    # The projection matrix is the intrisic matrix of the camera multiplied by the extrinsic matrix.
    # When the intrinsic matrix is the identity.
    # The results is the [ Rotation | translation ] in a 3x4 matrix

    point3D = cv2.triangulatePoints(P1, P2, pts_b.T, pts_a.T).T
    point3D = point3D[:, :3] / point3D[:, 3:4]

    # Return the triangulated 3D points
    # Return the position and orientation of the LH2-B wrt LH2-A
    return point3D

def scale_scene_to_real_size(df, calib_data):
    """
    Code takes the solved 3D scene and scales the scene so that the distance between the gridpoints is indeed 40mm

    --- Input
    df: dataframe with the triangulated position of the grid-points and the real position of the grid-points
    --- Output
    df: dataframe with the updated scaled-up scene
    """

    #
    # Slice the calibration data and add it to the  data dataframe.
    table_up = df.loc[ (df['timestamp'] > calib_data['timestamps_lh2']["table_up"][0]) & (df['timestamp'] < calib_data['timestamps_lh2']["table_up"][1])].mean(axis=0, numeric_only=True)
    tl = df.loc[ (df['timestamp'] > calib_data['timestamps_lh2']["tl"][0]) & (df['timestamp'] < calib_data['timestamps_lh2']["tl"][1])].mean(axis=0, numeric_only=True)
    tr = df.loc[ (df['timestamp'] > calib_data['timestamps_lh2']["tr"][0]) & (df['timestamp'] < calib_data['timestamps_lh2']["tr"][1])].mean(axis=0, numeric_only=True)
    bl = df.loc[ (df['timestamp'] > calib_data['timestamps_lh2']["bl"][0]) & (df['timestamp'] < calib_data['timestamps_lh2']["bl"][1])].mean(axis=0, numeric_only=True)
    br = df.loc[ (df['timestamp'] > calib_data['timestamps_lh2']["br"][0]) & (df['timestamp'] < calib_data['timestamps_lh2']["br"][1])].mean(axis=0, numeric_only=True)

    # Grab the point at (0,0,0) mm and (40,0,0) mm and use them to calibrate/scale the system.
    lh2_p1 = tr[['LH_x', 'LH_y', 'LH_z']].values
    lh2_p2 = bl[['LH_x', 'LH_y', 'LH_z']].values

    mocap_p1 = tr[['real_x_mm', 'real_y_mm', 'real_z_mm']].values
    mocap_p2 = bl[['real_x_mm', 'real_y_mm', 'real_z_mm']].values

    scale = np.linalg.norm(mocap_p2 - mocap_p1) / np.linalg.norm(lh2_p2 - lh2_p1) 
    # Scale all the points
    df['LH_x'] *= scale
    df['LH_y'] *= scale
    df['LH_z'] *= scale

    # Return scaled up scene
    return df, scale

def scale_scene_to_real_size_40cm(df):
    """
    Code takes the solved 3D scene and scales the scene so that the distance between the gridpoints is indeed 40mm

    --- Input
    df: dataframe with the triangulated position of the grid-points and the real position of the grid-points
    --- Output
    df: dataframe with the updated scaled-up scene
    """
    # Grab the point at (0,0,0) mm and (40,0,0) mm and use them to calibrate/scale the system.
    scale_p1 = df.loc[(df['real_x_mm'] == 0)  & (df['real_y_mm'] == 0) & (df['real_z_mm'] == 0), ['LH_x', 'LH_y', 'LH_z']].values.mean(axis=0)
    scale_p2 = df.loc[(df['real_x_mm'] == 40) & (df['real_y_mm'] == 0) & (df['real_z_mm'] == 0), ['LH_x', 'LH_y', 'LH_z']].values.mean(axis=0)
    scale = 40 / np.linalg.norm(scale_p2 - scale_p1)
    # Scale all the points
    df['LH_x'] *= scale
    df['LH_y'] *= scale
    df['LH_z'] *= scale

    # Return scaled up scene
    return df

def compute_distance_between_grid_points(df):
    """
    Code that calculates the mean error and std deviation of the distance between grid points.

    --- Input
    df: dataframe with the scaled triangulated position of the grid-points and the real position of the grid-points
    --- Output
    x_dist: array float - X-axis distances between adjacent grid-points 
    y_dist: array float - Y-axis distances between adjacent grid-points 
    z_dist: array float - Z-axis distances between adjacent grid-points 
    """

    ##################### GET X AXIS DISTANCES
    x_dist = []
    for y in [0, 40, 80, 120]:
        for z in [0, 40, 80, 120, 160]:
            for x in [0, 40, 80, 160, 200]:  # We are missing x=240 because we only want the distance between the points, not the actual points.
                # Grab all the points
                p1 = df.loc[(df['real_x_mm'] == x)  & (df['real_y_mm'] == y) & (df['real_z_mm'] == z), ['LH_x', 'LH_y', 'LH_z']].values
                p2 = df.loc[(df['real_x_mm'] == x+40)  & (df['real_y_mm'] == y) & (df['real_z_mm'] == z), ['LH_x', 'LH_y', 'LH_z']].values

                # Now permute all the distances between all the points in each position
                for v in p1:
                    for w in p2:
                        x_dist.append(np.linalg.norm(v-w))

    ##################### GET Y AXIS DISTANCES
    y_dist = []
    for y in [0, 40, 80]:        # We are missing y=120 because we only want the distance between the points, not the actual points.
        for z in [0, 40, 80, 120, 160]:
            for x in [0, 40, 80, 160, 200, 240]: 
                # Grab all the points
                p1 = df.loc[(df['real_x_mm'] == x)  & (df['real_y_mm'] == y) & (df['real_z_mm'] == z), ['LH_x', 'LH_y', 'LH_z']].values
                p2 = df.loc[(df['real_x_mm'] == x)  & (df['real_y_mm'] == y+40) & (df['real_z_mm'] == z), ['LH_x', 'LH_y', 'LH_z']].values

                # Now permute all the distances between all the points in each position
                for v in p1:
                    for w in p2:
                        y_dist.append(np.linalg.norm(v-w))

    ##################### GET Z AXIS DISTANCES
    z_dist = []
    for y in [0, 40, 80, 120]:
        for z in [0, 40, 80, 120]:       # We are missing z=160 because we only want the distance between the points, not the actual points.
            for x in [0, 40, 80, 160, 200, 240]: 
                # Grab all the points
                p1 = df.loc[(df['real_x_mm'] == x)  & (df['real_y_mm'] == y) & (df['real_z_mm'] == z), ['LH_x', 'LH_y', 'LH_z']].values
                p2 = df.loc[(df['real_x_mm'] == x)  & (df['real_y_mm'] == y) & (df['real_z_mm'] == z+40), ['LH_x', 'LH_y', 'LH_z']].values

                # Now permute all the distances between all the points in each position
                for v in p1:
                    for w in p2:
                        z_dist.append(np.linalg.norm(v-w))

    # At the end, put all the distances together in an array and calculate mean and std
    x_dist = np.array(x_dist)
    y_dist = np.array(y_dist)
    z_dist = np.array(z_dist)
    # Remove ouliers, anything bigger than 1 meters gets removed.
    x_dist = x_dist[x_dist <= 500]
    y_dist = y_dist[y_dist <= 500]
    z_dist = z_dist[z_dist <= 500]

    return x_dist, y_dist, z_dist

def correct_perspective(df, calib_data):
    """
    Create a rotation and translation vector to move the reconstructed grid onto the origin for better comparison.
    Using an SVD, according to: https://nghiaho.com/?page_id=671
    """

    # Get the calibration corners
    table_up = df.loc[ (df['timestamp'] > calib_data['timestamps_lh2']["table_up"][0]) & (df['timestamp'] < calib_data['timestamps_lh2']["table_up"][1])].mean(axis=0, numeric_only=True)
    tl = df.loc[ (df['timestamp'] > calib_data['timestamps_lh2']["tl"][0]) & (df['timestamp'] < calib_data['timestamps_lh2']["tl"][1])].mean(axis=0, numeric_only=True)
    tr = df.loc[ (df['timestamp'] > calib_data['timestamps_lh2']["tr"][0]) & (df['timestamp'] < calib_data['timestamps_lh2']["tr"][1])].mean(axis=0, numeric_only=True)
    bl = df.loc[ (df['timestamp'] > calib_data['timestamps_lh2']["bl"][0]) & (df['timestamp'] < calib_data['timestamps_lh2']["bl"][1])].mean(axis=0, numeric_only=True)
    br = df.loc[ (df['timestamp'] > calib_data['timestamps_lh2']["br"][0]) & (df['timestamp'] < calib_data['timestamps_lh2']["br"][1])].mean(axis=0, numeric_only=True)
    
    # # Make an array with the ground truth target
    # B = np.array([table_up[['real_x_mm', 'real_y_mm', 'real_z_mm']],\
    #                     tl[['real_x_mm', 'real_y_mm', 'real_z_mm']],\
    #                     tr[['real_x_mm', 'real_y_mm', 'real_z_mm']],\
    #                     bl[['real_x_mm', 'real_y_mm', 'real_z_mm']],\
    #                     br[['real_x_mm', 'real_y_mm', 'real_z_mm']]])
    
    # # Make an array with the LH reconstructed points
    # A = np.array([table_up[['LH_x', 'LH_y', 'LH_z']],\
    #                     tl[['LH_x', 'LH_y', 'LH_z']],\
    #                     tr[['LH_x', 'LH_y', 'LH_z']],\
    #                     bl[['LH_x', 'LH_y', 'LH_z']],\
    #                     br[['LH_x', 'LH_y', 'LH_z']]])
    
    # Make an array with the ground truth target
    B = np.array([      tl[['real_x_mm', 'real_y_mm', 'real_z_mm']],\
                        tr[['real_x_mm', 'real_y_mm', 'real_z_mm']],\
                        bl[['real_x_mm', 'real_y_mm', 'real_z_mm']],\
                        br[['real_x_mm', 'real_y_mm', 'real_z_mm']]])
    
    # Make an array with the LH reconstructed points
    A = np.array([      tl[['LH_x', 'LH_y', 'LH_z']],\
                        tr[['LH_x', 'LH_y', 'LH_z']],\
                        bl[['LH_x', 'LH_y', 'LH_z']],\
                        br[['LH_x', 'LH_y', 'LH_z']]])

    # B = np.unique(df[['real_x_mm','real_y_mm','real_z_mm']].to_numpy(), axis=0)
    # A = np.empty_like(B, dtype=float)
    # for i in range(B.shape[0]):
    #     A[i] = df.loc[(df['real_x_mm'] == B[i,0])  & (df['real_y_mm'] == B[i,1]) & (df['real_z_mm'] == B[i,2]), ['LH_x', 'LH_y', 'LH_z']].values.mean(axis=0)

    # Get  all the reconstructed points
    A2 = df[['LH_x','LH_y','LH_z']].to_numpy().T

    # Convert the point to column vectors,
    # to match twhat the SVD algorithm expects
    A = A.T
    B = B.T

    # Get the centroids
    A_centroid = A.mean(axis=1).reshape((-1,1))
    B_centroid = B.mean(axis=1).reshape((-1,1))

    # Get H
    H = (A - A_centroid) @ (B - B_centroid).T

    # Do the SVD
    U, S, V = np.linalg.svd(H)

    # Get the rotation matrix
    R = V @ U.T

    # check for errors, and run the correction
    if np.linalg.det(R) < 0:
        U, S, V = np.linalg.svd(R)
        V[:,2] = -1*V[:,2]
        R = V @ U.T


    # Override the rotation matrix with the montecarlo value

    R = np.array([[-0.49281973, -0.85385419, -0.16751638],
                  [ 0.86817948, -0.46962783, -0.16035615],
                  [ 0.05825042, -0.22446096,  0.97274055]])

    # Get the ideal translation
    t = B_centroid - R @ A_centroid

    correct_points = (R@A2 + t)
    correct_points = correct_points.T

    # Update dataframe
    df['Rt_x'] = correct_points[:,0]
    df['Rt_y'] = correct_points[:,1]
    df['Rt_z'] = correct_points[:,2]
    return df, R, t

def correct_perspective_40cm(df):
    """
    Create a rotation and translation vector to move the reconstructed grid onto the origin for better comparison.
    Using an SVD, according to: https://nghiaho.com/?page_id=671
    """
    
    B = np.unique(df[['real_x_mm','real_y_mm','real_z_mm']].to_numpy(), axis=0)
    A = np.empty_like(B, dtype=float)
    for i in range(B.shape[0]):
        A[i] = df.loc[(df['real_x_mm'] == B[i,0])  & (df['real_y_mm'] == B[i,1]) & (df['real_z_mm'] == B[i,2]), ['LH_x', 'LH_y', 'LH_z']].values.mean(axis=0)

    # Get  all the reconstructed points
    A2 = df[['LH_x','LH_y','LH_z']].to_numpy().T

    # Convert the point to column vectors,
    # to match twhat the SVD algorithm expects
    A = A.T
    B = B.T

    # Get the centroids
    A_centroid = A.mean(axis=1).reshape((-1,1))
    B_centroid = B.mean(axis=1).reshape((-1,1))

    # Get H
    H = (A - A_centroid) @ (B - B_centroid).T

    # Do the SVD
    U, S, V = np.linalg.svd(H)

    # Get the rotation matrix
    R = V @ U.T

    # check for errors, and run the correction
    if np.linalg.det(R) < 0:
        U, S, V = np.linalg.svd(R)
        V[:,2] = -1*V[:,2]
        R = V @ U.T

    # Get the ideal translation
    t = B_centroid - R @ A_centroid

    correct_points = (R@A2 + t)
    correct_points = correct_points.T

    # Update dataframe
    df['Rt_x'] = correct_points[:,0]
    df['Rt_y'] = correct_points[:,1]
    df['Rt_z'] = correct_points[:,2]
    return df

def compute_errors(df):
    """Calculate MAE, RMS and Precision for a particular reconstruction"""
    # Extract needed data from the main dataframe
    points = df[['Rt_x','Rt_y','Rt_z']].to_numpy()
    ground_truth = df[['real_x_mm','real_y_mm','real_z_mm']].to_numpy()

    # Calculate distance between points and their ground truth
    errors =  np.linalg.norm(ground_truth - points, axis=1)
    # print the mean and standard deviation
    mae = errors.mean()
    rmse = np.sqrt((errors**2).mean())
    std = errors.std()

    return mae, rmse, std

def is_coplanar(points):
    """
    taken from the idea here: https://stackoverflow.com/a/72384583
    returns True or False depending if the points are too coplanar or not.
    """

    best_fit = Plane.best_fit(points)
    distances = np.empty((points.shape[0]))

    for i in range(points.shape[0]):
        distances[i] = best_fit.distance_point(points[i])

    error = distances.mean()
    return error

def compute_mad(points):
    """ Get a list of 3d points and calculate the Median Absolute Deviation"""

    centroid = points.mean(axis=0)
    return np.linalg.norm(points - centroid, axis=1).mean()

def max_distance(points):
    """
    Calculate the maximum Euclidean distance between any two points in a 3D space.

    Parameters:
    points (numpy.ndarray): An array of points in 3D space.

    Returns:
    float: The maximum distance found between any two points.
    """
    centroid = points.mean(axis=0)
    return np.linalg.norm(points - centroid, axis=1).max()


def back_propagate_3D_point(point, R_star, t_star, scale, R, t):
    """
    Grab a real world 3D point and back calculate where in the LH projection image it would be seen.
    """
    # Check if the 3D point is in the correct shape: column vector (3,1)
    if point.shape == (3,):
        point = point[:,np.newaxis]
    if point.shape == (1,3):
        point = point.T

    # undo the rigid body transformation and de-scale the scene
    lh_point = R.T @ (point - t) / scale

    # Make the 3D point into homogeneous coordinates, and remember you flipped Y and Z because of discordances between the groundtruth and the LH data coordinate axes.
    lh_point_hom = np.array([lh_point[0,0], lh_point[2,0], lh_point[1,0], 1]).reshape((-1,1))

    # Reconstruct the projection matrices
    R_1 = np.eye(3,dtype='float64')
    t_1 = np.zeros((3,1),dtype='float64')
    P1 = np.hstack([R_1.T, -R_1.T.dot(t_1)])
    P2 = np.hstack([R_star.T, -R_star.T.dot(t_star)])

    # Project the 3D point
    LHA_proj_point = P1 @ lh_point_hom
    LHC_proj_point = P2 @ lh_point_hom

    # Transform point back to (u,v) from homogeneus coordinates.
    LHA_proj_point = LHA_proj_point[0:2] / LHA_proj_point[2]
    LHC_proj_point = LHC_proj_point[0:2] / LHC_proj_point[2]

    return [LHA_proj_point, LHC_proj_point]


def forward_propagate_2D_point(pointA, pointB, R_star, t_star, scale, R, t):
    """
    Grab 2 observations from the LH and estimate the 3D reconstruction.
    """
    # Check if the 2D points are in the correct shape: column vector (2,1)
    if pointA.shape == (2,):
        pointA = pointA[:,np.newaxis]
    if pointA.shape == (1,2):
        pointA = pointA.T

    if pointB.shape == (2,):
        pointB = pointB[:,np.newaxis]
    if pointB.shape == (1,2):
        pointB = pointB.T

    # pointA = array([[0.06056018], [0.20392456]])  shape = (2,1)
    # pointB = array([[-0.04422776], [0.35725837]])  shape = (2,1)

    # Reconstruct the projection matrices
    R_1 = np.eye(3,dtype='float64')
    t_1 = np.zeros((3,1),dtype='float64')
    P1 = np.hstack([R_1.T, -R_1.T.dot(t_1)])
    P2 = np.hstack([R_star.T, -R_star.T.dot(t_star)])

    # P1 = array([[ 1.,  0.,  0., -0.],
    #             [ 0.,  1.,  0., -0.],
    #             [ 0.,  0.,  1., -0.]])

    # P2 = array([[ 0.64253069, -0.204868  , -0.73836537,  0.90646167],
    #             [ 0.36707223,  0.92813002,  0.06190841,  0.11865332],
    #             [ 0.67261601, -0.31081148,  0.67155337,  0.40527599]])

    # triangulate the point
    point3D = cv2.triangulatePoints(P1, P2, pointA, pointB).T

    # point3D = array([[-0.04615691, -0.1512819 , -0.78239695, -0.60236064]])

    # Undo the homogeneous coordinates
    point3D = point3D[:, :3] / point3D[:, 3:4]
    ##### TODO, code diverges here
    # point3D = array([[0.07662671, 0.25114839, 1.2988846]])
    # remember you flipped Y and Z because of discordances between the groundtruth and the LH data coordinate axes.
    point3D = point3D[:,[0,2,1]]
    # point3D = array([[0.07662671, 1.2988846 , 0.25114839]])

    # Scale up the triangulate to real world scale
    point3D *= scale
    # point3D = array([[ 114.81827479, 1946.26250823,  376.32342121]])
    # do the rigid body transformation
    point3D = R@point3D.T + t
    # point3D = array( [[-616.19159585],
    #                   [1186.59377601],
    #                   [1032.48006645]])
    return point3D