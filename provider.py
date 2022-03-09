import numpy as np


def normalize_data(batch_data):
    """ Normalize the batch data, use coordinates of the block centered at origin,
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    B, N, C = batch_data.shape
    normal_data = np.zeros((B, N, C))
    for b in range(B):
        pc = batch_data[b]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        normal_data[b] = pc
    return normal_data


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx

def shuffle_points(batch_data):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    idx = np.arange(batch_data.shape[1])
    np.random.shuffle(idx)
    return batch_data[:,idx,:]

# def rotate_point_cloud(batch_data):
#     """ Randomly rotate the point clouds to augument the dataset
#         rotation is per shape based along up direction
#         Input:
#           BxNx3 array, original batch of point clouds
#         Return:
#           BxNx3 array, rotated batch of point clouds
#     """
#     rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
#     for k in range(batch_data.shape[0]):
#         rotation_angle = np.random.uniform() * 2 * np.pi
#         cosval = np.cos(rotation_angle)
#         sinval = np.sin(rotation_angle)
#         rotation_matrix = np.array([[cosval, 0, sinval],
#                                     [0, 1, 0],
#                                     [-sinval, 0, cosval]])
#         shape_pc = batch_data[k, ...]
#         rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
#     return rotated_data

def rotate_point_cloud_z(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def rotate_point_cloud_with_normal(batch_xyz_normal):
    ''' Randomly rotate XYZ, normal point cloud.
        Input:
            batch_xyz_normal: B,N,6, first three channels are XYZ, last 3 all normal
        Output:
            B,N,6, rotated XYZ, normal point cloud
    '''
    for k in range(batch_xyz_normal.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_xyz_normal[k,:,0:3]
        shape_normal = batch_xyz_normal[k,:,3:6]
        batch_xyz_normal[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        batch_xyz_normal[k,:,3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return batch_xyz_normal

def rotate_perturbation_point_cloud_with_normal(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx6 array, original batch of point clouds and point normals
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
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
        shape_pc = batch_data[k,:,0:3]
        shape_normal = batch_data[k,:,3:6]
        rotated_data[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), R)
        rotated_data[k,:,3:6] = np.dot(shape_normal.reshape((-1, 3)), R)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k,:,0:3]
        rotated_data[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle_with_normal(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx6 array, original batch of point clouds with normal
          scalar, angle of rotation
        Return:
          BxNx6 array, rotated batch of point clouds iwth normal
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k,:,0:3]
        shape_normal = batch_data[k,:,3:6]
        rotated_data[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data[k,:,3:6] = np.dot(shape_normal.reshape((-1,3)), rotation_matrix)
    return rotated_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
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
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data


def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data


def random_scale_point_cloud(batch_data, scale_inter):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    scale_high = 1 + scale_inter
    scale_low = 1 / scale_high
    B, N, C = batch_data.shape

    scale_save = np.zeros([3, B])
    for batch_index in range(B):
        scales = np.random.uniform(scale_low, scale_high, 3)
        S = np.diag(scales)
        for k in range(3):
            scale_save[k, batch_index] = scales[k]
        batch_data[batch_index, :, :] = np.dot(batch_data[batch_index, :, :], S)

    return batch_data, scale_save


# def generate_scale_matrix(scales):
def random_rotate_point_cloud(batch_data, angle_inter=0.3):
    # sr_data = np.zeros(batch_data.shape, dtype=np.float32)
    B = batch_data.shape[0]
    # N = batch_data.shape[1]
    angles_save = np.zeros([3, B])
    # trans = np.zeros([B, 3, 3], dtype=np.float32)
    for batch_index in range(B):
        angles = angle_inter * (2 * np.random.rand(3) - 1)
        r_angles = angles * np.pi
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(r_angles[0]), -np.sin(r_angles[0])],
                       [0, np.sin(r_angles[0]), np.cos(r_angles[0])]])
        Ry = np.array([[np.cos(r_angles[1]), 0, np.sin(r_angles[1])],
                       [0, 1, 0],
                       [-np.sin(r_angles[1]), 0, np.cos(r_angles[1])]])
        Rz = np.array([[np.cos(r_angles[2]), -np.sin(r_angles[2]), 0],
                       [np.sin(r_angles[2]), np.cos(r_angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        for k in range(3):
            angles_save[k, batch_index] = angles[k]
        batch_data[batch_index, :, :] = np.dot(batch_data[batch_index, :, :], R)
        # angles_save[batch_index][batch_index, :] = angles
        # shape_pc = batch_data[k, ...]
        # sr_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)

    return batch_data, angles_save


def rotate_point_cloud(batch_data, angles, axis):
    # sr_data = np.zeros(batch_data.shape, dtype=np.float32)
    B,N,_ = batch_data.shape
    # trans = np.zeros([B, 3, 3], dtype=np.float32)
    r_angles = angles * np.pi

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r_angles), -np.sin(r_angles)],
                   [0, np.sin(r_angles), np.cos(r_angles)]])
    Ry = np.array([[np.cos(r_angles), 0, np.sin(r_angles)],
                   [0, 1, 0],
                   [-np.sin(r_angles), 0, np.cos(r_angles)]])
    Rz = np.array([[np.cos(r_angles), -np.sin(r_angles), 0],
                   [np.sin(r_angles), np.cos(r_angles), 0],
                   [0, 0, 1]])
    # R = np.dot(Rz, np.dot(Ry, Rx))
    if axis == 0:
        R = Rx
    elif axis == 1:
        R = Ry
    elif axis == 2:
        R = Rz
    else:
        R = np.dot(Rz, np.dot(Ry, Rx))
    for batch_index in range(B):
        batch_data[batch_index, :, :] = np.dot(batch_data[batch_index, :, :], R)
        # angles_save[batch_index][batch_index, :] = angles
        # shape_pc = batch_data[k, ...]
        # sr_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)

    return batch_data


def scale_point_cloud(batch_data, scales, axis):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B = batch_data.shape[0]
    if axis == 0:
        S = np.diag(np.array([scales, 1, 1]))
    elif axis == 1:
        S = np.diag(np.array([1, scales, 1]))
    elif axis == 2:
        S = np.diag(np.array([1, 1, scales]))
    else:
        S = np.diag(np.array([scales, scales, scales]))

    for batch_index in range(B):
        batch_data[batch_index, :, :] = np.dot(batch_data[batch_index, :, :], S)
    return batch_data


def generate_rotate_matrix(angles):
    B = angles.shape[1]
    rotation = np.zeros((B, 3, 3))
    for i in range(B):
        r_angles = angles[:, i] * np.pi
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(r_angles[0]), -np.sin(r_angles[0])],
                       [0, np.sin(r_angles[0]), np.cos(r_angles[0])]])
        Ry = np.array([[np.cos(r_angles[1]), 0, np.sin(r_angles[1])],
                       [0, 1, 0],
                       [-np.sin(r_angles[1]), 0, np.cos(r_angles[1])]])
        Rz = np.array([[np.cos(r_angles[2]), -np.sin(r_angles[2]), 0],
                       [np.sin(r_angles[2]), np.cos(r_angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        rotation[i, :, :] = R
    return rotation


def generate_a_rotate_matrix(angles):
    # rotation = np.zeros((3, 3))
    r_angles = angles * np.pi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r_angles[0]), -np.sin(r_angles[0])],
                   [0, np.sin(r_angles[0]), np.cos(r_angles[0])]])
    Ry = np.array([[np.cos(r_angles[1]), 0, np.sin(r_angles[1])],
                   [0, 1, 0],
                   [-np.sin(r_angles[1]), 0, np.cos(r_angles[1])]])
    Rz = np.array([[np.cos(r_angles[2]), -np.sin(r_angles[2]), 0],
                   [np.sin(r_angles[2]), np.cos(r_angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc


def random_sr_point_cloud(batch_data, scale_inter=0.3, angle_inter=0.3):
    sr_data = np.zeros(batch_data.shape, dtype=np.float32)
    B = batch_data.shape[0]
    # N = batch_data.shape[1]
    trans = np.zeros([B, 3, 3], dtype=np.float32)
    for k in range(B):
        angles = angle_inter * (2 * np.random.rand(3) - 1) * np.pi
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
        shape_pc = batch_data[k, ...]
        sr_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)


        scales = scale_inter * (2 * np.random.rand(3) - 1) + 1
        S = np.diag(scales)
        # S = np.tile(scales.reshape((1, 3)), (N, 1))
        sr_data[k, ...] = np.dot(sr_data[k, ...], S)
        # trans[k, :, 3:4] = scales.reshape((3, 1))
        trans[k, :, :] = np.dot(R, S)
    return sr_data, trans


if __name__ == '__main__':
    scale_save = {i: np.zeros([2, 3]) for i in range(3)}
    print(scale_save)





