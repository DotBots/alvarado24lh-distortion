import pandas as pd
import numpy as np
from functions.data_processing import   read_dataset_files, \
                                        LH2_count_to_pixels, \
                                        mocap_to_pixels, \
                                        scale_scene_to_real_size, \
                                        compute_distance_between_grid_points, \
                                        back_propagate_3D_point, \
                                        forward_propagate_2D_point, \
                                        correct_perspective

from functions.plotting import plot_transformed_3D_data, \
                                plot_projected_LH_views

#############################################################################
###                                Options                                ###
#############################################################################

# file with the data to analyze
# LH data files
lh_file = './dataset/lh2/LHB-DotBox/lh_data.csv'
lh_file = './dataset/lh2/LHC-DotBox/lh_data.csv'

# Mocap data file
mocap_file = './dataset/mocap/LHB-DotBox/mocap_data.csv'
mocap_file = './dataset/mocap/LHC-DotBox/mocap_data.csv'

## TODO Add mocap file here and calib file.

#############################################################################
###                                  Main                                 ###
#############################################################################

if __name__ == "__main__":

    # Import data
    # df=pd.read_csv(data_file, index_col=0)
    # df, calib_data = read_dataset_files(lh_file, mocap_file, calib_file)
    df, basestation_pose = read_dataset_files(lh_file, mocap_file)

    # Project sweep angles on to the z=1 image plane
    pts_lighthouse = LH2_count_to_pixels(df['lfsr_index_0'].values, df['lfsr_index_1'].values, 0)

    # Add the LH2 projected matrix into the dataframe that holds the info about what point is where in real life.
    df['LH_proj_x'] = pts_lighthouse[:,0]
    df['LH_proj_y'] = pts_lighthouse[:,1]

    # Mocap to pixels
    mocap_pixels = mocap_to_pixels(df[['dotbot_x_mm', 'dotbot_y_mm', 'dotbot_z_mm']].values, basestation_pose)









    # Add The 3D point to the Dataframe that has the real coordinates, timestamps etc.
    # This will help correlate which point are supposed to go where.
    df['LH_x'] = point3D[:,0]
    df['LH_y'] = point3D[:,2]   # We need to invert 2 of the axis because the LH2 frame Z == depth and Y == Height
    df['LH_z'] = point3D[:,1]   # But the dataset assumes X = Horizontal, Y = Depth, Z = Height

    # Scale the scene to real size
    df, scale = scale_scene_to_real_size(df, calib_data)

    # Bring reconstructed data to the origin for easier comparison
    df, R, t = correct_perspective(df, calib_data)

    # Compute distances between gridpoints
    # x_dist, y_dist, z_dist = compute_distance_between_grid_points(df)
    point = df[['real_x_mm', 'real_y_mm', 'real_z_mm']].iloc[0].values
    LHA_top_point, LHC_top_point = back_propagate_3D_point(point, R_star, t_star, scale, R, t)
    LHA_top_point = df[['LHA_proj_x', 'LHA_proj_y']].iloc[2].values
    LHC_top_point = df[['LHB_proj_x', 'LHB_proj_y']].iloc[2].values
    top_point_3D = forward_propagate_2D_point(LHA_top_point, LHC_top_point, R_star, t_star, scale, R, t)

    #############################################################################
    ###                             Plotting                                  ###
    #############################################################################

    # Plot data
    plot_projected_LH_views(pts_lighthouse_A, pts_lighthouse_B, [LHA_top_point, LHC_top_point])

    # Plot superimposed "captured data" vs. "ground truth"
    plot_transformed_3D_data(df)

