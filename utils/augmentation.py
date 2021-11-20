import numpy as np


def random_rotate_point_cloud(data):
    """
    :param data: Nx3 array
    :return: rotatation_matrix: Nx3 array
    """
    angles = [np.random.uniform() * 2 * np.pi, np.random.uniform() * 2 * np.pi, np.random.uniform() * 2 * np.pi]
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    rotated_data = np.dot(data, R)

    return rotated_data


def jitter_point_cloud(data, sigma=0.01, clip=0.05):
    """
    :param data: Nx3 array
    :return: jittered_data: Nx3 array
    """
    N, C = data.shape
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    jittered_data += data

    return jittered_data


def random_scale_point_cloud(data, scale_low=0.8, scale_high=1.25):
    """
    :param data:  Nx3 array
    :return: scaled_data:  Nx3 array
    """
    scale = np.random.uniform(scale_low, scale_high)
    scaled_data = data * scale

    return scaled_data


def random_dropout_point_cloud(data, p=0.9):
    """
    :param data:  Nx3 array
    :return: dropout_data:  Nx3 array
    """
    N, C = data.shape
    dropout_ratio = np.random.random() * p
    drop_idx = np.where(np.random.random(N) <= dropout_ratio)[0]
    dropout_data = np.zeros_like(data)
    if len(drop_idx) > 0:
        dropout_data[drop_idx, :] = data[0, :]

    return dropout_data


######### SONET ########

def rotate_point_cloud_90(data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    rotated_data = np.zeros(data.shape, dtype=np.float32)

    rotation_angle = np.random.randint(low=0, high=4) * (np.pi/2.0)
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(data.reshape((-1, 3)), rotation_matrix)

    return rotated_data


def rotate_point_cloud(data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    rotated_data = np.zeros(data.shape, dtype=np.float32)

    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(data.reshape((-1, 3)), rotation_matrix)

    return rotated_data


def rotate_point_cloud_with_normal_som(pc, surface_normal, som):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """

    rotation_angle = np.random.uniform() * 2 * np.pi
    # rotation_angle = np.random.randint(low=0, high=12) * (2*np.pi / 12.0)
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])

    rotated_pc = np.dot(pc, rotation_matrix)
    rotated_surface_normal = np.dot(surface_normal, rotation_matrix)
    rotated_som = np.dot(som, rotation_matrix)

    return rotated_pc, rotated_surface_normal, rotated_som


def rotate_perturbation_point_cloud(data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
                   [0,np.cos(angles[0]),-np.sin(angles[0])],
                   [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                   [0,1,0],
                   [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                   [np.sin(angles[2]),np.cos(angles[2]),0],
                   [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))

    rotated_data = np.dot(data, R)

    return rotated_data


def rotate_perturbation_point_cloud_with_normal_som(pc, surface_normal, som, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """

    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
                   [0,np.cos(angles[0]),-np.sin(angles[0])],
                   [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                   [0,1,0],
                   [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                   [np.sin(angles[2]),np.cos(angles[2]),0],
                   [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))

    rotated_pc = np.dot(pc, R)
    rotated_surface_normal = np.dot(surface_normal, R)
    rotated_som = np.dot(som, R)

    return rotated_pc, rotated_surface_normal, rotated_som
