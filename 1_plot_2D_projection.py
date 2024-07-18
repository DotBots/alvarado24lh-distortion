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
# lh_file = './dataset/lh2/LHB-DotBox/lh_data.csv'
lh_file = './dataset/lh2/LHC-DotBox/lh_data.csv'

# Mocap data file
# mocap_file = './dataset/mocap/LHB-DotBox/mocap_data.csv'
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

    # Add to the dataframe
    df['Mocap_proj_x'] = mocap_pixels[:,0]
    df['Mocap_proj_y'] = mocap_pixels[:,1]

    #############################################################################
    ###                             Plotting                                  ###
    #############################################################################

    # Plot data
    n = -1
    print(df.iloc[0])
    print(df.iloc[4000])
    plot_projected_LH_views(pts_lighthouse[0:n], mocap_pixels[0:n])
    # plot_projected_LH_views(pts_lighthouse, mocap_pixels)

    # # Plot superimposed "captured data" vs. "ground truth"
    # plot_transformed_3D_data(df)

