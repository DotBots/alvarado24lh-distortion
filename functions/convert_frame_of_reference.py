import numpy as np
import math

def deg_to_rad(degrees):
    return degrees * math.pi / 180

def rotation_matrix(roll, pitch, yaw):
    # Convert angles from degrees to radians
    roll = math.radians(roll)
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)

    # Create individual rotation matrices
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    # Combined rotation matrix. The order is ZYX because we first apply yaw, then pitch, then roll
    R = np.dot(R_z, np.dot(R_y, R_x))

    return R

def transform_coordinates(obj1_pos, obj1_orientation, obj2_pos):
    # Inverse rotation matrix for the first object
    R_inv = np.linalg.inv(rotation_matrix(*obj1_orientation))

    # Translate second object's position to the origin of the first object
    translated_pos = np.array(obj2_pos) - np.array(obj1_pos)

    # Rotate the translated position
    transformed_pos = np.dot(R_inv, translated_pos)

    return transformed_pos

def cartesian_to_spherical(cartesian_coords):
    x, y, z = cartesian_coords
    r = math.sqrt(x**2 + y**2 + z**2)  # Radial distance
    theta = math.pi/2 - math.acos(-y / r)  # Polar angle
    phi = math.atan2(z, x)    # Azimuthal angle
    return r, theta, phi

def LH2_angles_to_pixels(azimuth, elevation):
    """
    Project the Azimuth and Elevation angles of a LH2 basestation into the unit image plane.
    """
    pts_lighthouse = np.array([np.tan(azimuth),         # horizontal pixel  
                               np.tan(elevation) * 1/np.cos(azimuth)]).T    # vertical   pixel 
    return pts_lighthouse

def create_projection_matrix(position, orientation):
    # Rotation matrix
    R = rotation_matrix(*orientation)

    # Translation vector (camera position)
    t = -np.array(position).reshape(3, 1)

    # Extrinsic matrix [R | -R*t]
    extrinsic_matrix = np.concatenate((R, np.dot(R, t)), axis=1)

    # Intrinsic matrix is the identity matrix in this case
    intrinsic_matrix = np.identity(3)

    # Projection matrix P = K * [R | -R*t]
    projection_matrix = np.dot(intrinsic_matrix, extrinsic_matrix)

    return projection_matrix

# Object 1 position and orientation (roll, pitch, yaw)
LHA_pos = [1078.64, 2093.3, 1232.8]
LHA_orientation = [93.28, 28.47, -166.81]
LHC_pos = [1210.25, 622.53, 1351.9]
LHC_orientation = [87.12, -17.70, -151.91]

# Object 2 position
DB_pos = [-596.42, 1182.43, 1089.39]

# Transform Object 2's coordinates into the frame of reference of Object 1
DB_lha = transform_coordinates(LHA_pos, LHA_orientation, DB_pos)
DB_lhc = transform_coordinates(LHC_pos, LHC_orientation, DB_pos)
print(f"DB_lha = {DB_lha}")
print(f"DB_lhc = {DB_lhc}")

# Transform to spherical coordinates.
DB_lha_sphe = cartesian_to_spherical(DB_lha)
DB_lhc_sphe = cartesian_to_spherical(DB_lhc)
print(f"DB_lha_sphe = {DB_lha_sphe}")
print(f"DB_lhc_sphe = {DB_lhc_sphe}")

DB_lha_pix = LH2_angles_to_pixels(DB_lha_sphe[1], DB_lha_sphe[2])
DB_lhc_pix = LH2_angles_to_pixels(DB_lhc_sphe[1], DB_lhc_sphe[2])
print(f"DB_lha_pix = {DB_lha_pix}")
print(f"DB_lhc_pix = {DB_lhc_pix}")

## Calculate the LH projection matrices to calculate the groundtruth value of the top point
LHA_P = np.array([  [-8.55875944e-01, -4.76417833e-01, -2.01252124e-01, 2.16857110e+03],
                    [-2.00586300e-01, -5.28892190e-02,  9.78247344e-01, -8.78909917e+02],
                    [-4.76698547e-01,  8.77626788e-01, -5.02962803e-02, -1.26094478e+03]])

LHC_P = create_projection_matrix(LHC_pos, LHC_orientation)

DB_pos_hom = np.array([[-596.42, 1182.43, 1089.39, 1]]).T

DB_lha_pix = LHA_P @ DB_pos_hom
DB_lha_pix = DB_lha_pix[0:2] / DB_lha_pix[2]

DB_lhc_pix = LHC_P @ DB_pos_hom
DB_lhc_pix = DB_lhc_pix[0:2] / DB_lhc_pix[2]


a=5

