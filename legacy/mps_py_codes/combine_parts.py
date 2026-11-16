
import h5py
import numpy as np
import time

def combine_parts(base_filename, m = 512):
    # Define the size of the final cube
    """
    Combine the 12 parts of a channel flow simulation into a single 3D cube.

    Parameters
    ----------
    base_filename : str
        The base filename of the parts, without the part number.
    m : int
        The size of each side of the cube, in terms of the number of parts.
        The default value is 512, which corresponds to a cube size of
        2048x512x1536.

    Returns
    -------
    u, v, w : array_like
        The u, v, and w components of the velocity, respectively, as 3D
        arrays of shape (cube_size_x, cube_size_y, cube_size_z), where
        cube_size_x = 4*m, cube_size_y = m, and cube_size_z = 3*m.

    Notes
    -----
    This function assumes that the parts are labeled as
    <base_filename>_P1.h5, <base_filename>_P2.h5, ..., <base_filename>_P12.h5.
    The function also assumes that the data is stored in a dataset called
    'Velocity_0001' in each part file.

    The function prints the time taken to combine the parts to the console.
    """

    start_time = time.time()
    # m = 4
    cube_size_x = 4*m
    cube_size_y = m
    cube_size_z = 3*m

    # Preallocate the final cube arrays
    u = np.zeros((cube_size_z, cube_size_y, cube_size_x))
    v = np.zeros((cube_size_z, cube_size_y, cube_size_x))
    w = np.zeros((cube_size_z, cube_size_y, cube_size_x))

    # Define the parts and their corresponding indices
    parts = [
        (1, m,              1, m,   1, m),                 # Part 1
        (1, m,              1, m,   m + 1, 2 * m),         # Part 2
        (1, m,              1, m,   2 * m + 1, 3 * m),     # Part 3
        (m + 1, 2 * m,      1, m,   1, m),                 # Part 4
        (m + 1, 2 * m,      1, m,   m + 1, 2 * m),         # Part 5
        (m + 1, 2 * m,      1, m,   2 * m + 1, 3 * m),     # Part 6
        (2 * m + 1, 3 * m,  1, m,   1, m),                 # Part 7
        (2 * m + 1, 3 * m,  1, m,   m + 1, 2 * m),         # Part 8
        (2 * m + 1, 3 * m,  1, m,   2 * m + 1, 3 * m),     # Part 9
        (3 * m + 1, 4 * m,  1, m,   1, m),                 # Part 10
        (3 * m + 1, 4 * m,  1, m,   m + 1, 2 * m),         # Part 11
        (3 * m + 1, 4 * m,  1, m,   2 * m + 1, 3 * m),     # Part 12
    ]

    # Determine the variable name based on the base_filename
    variable_name_map = {
        'U_t1': 'Velocity_0001',
        'U_t1301': 'Velocity_1301',
        'U_t2601': 'Velocity_2601',
        'U_t3901': 'Velocity_3901'
    }
    
    variable_name = variable_name_map.get(base_filename)
    if not variable_name:
        raise ValueError('Unsupported base filename')

    # Loop over each part and insert it into the big cube
    for i, (xstart, xend, ystart, yend, zstart, zend) in enumerate(parts):
        part_file = f'Channelflow/3D test/{base_filename}_P{i+1}.h5'
        # part_file = f'Channelflow/3D/31 Grid 2048x512x1536_h5/{base_filename}_P{i+1}.h5'
        
        with h5py.File(part_file, 'r') as f:
            # part_data = f[variable_name][:]

            # # Insert the part data into the big cube
            # u[zstart-1:zend, ystart-1:yend, xstart-1:xend] = part_data[..., 0]
            # v[zstart-1:zend, ystart-1:yend, xstart-1:xend] = part_data[..., 1]
            # w[zstart-1:zend, ystart-1:yend, xstart-1:xend] = part_data[..., 2]

            # part_data = f[variable_name][:]
            velocity_dataset = f[variable_name]   
            vel = velocity_dataset[()]

            # print(xstart, xend, ystart, yend, zstart, zend)
            # Insert the part data into the big cube
            u[zstart-1:zend, ystart-1:yend, xstart-1:xend] = vel[:, :, :, 0]
            v[zstart-1:zend, ystart-1:yend, xstart-1:xend] = vel[:, :, :, 1]
            w[zstart-1:zend, ystart-1:yend, xstart-1:xend] = vel[:, :, :, 2]

    u = u.reshape((u.shape[2],u.shape[1],u.shape[0]))
    v = v.reshape((u.shape[2],u.shape[1],u.shape[0]))
    w = w.reshape((u.shape[2],u.shape[1],u.shape[0]))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Time taken to combine {base_filename}: {elapsed_time:.2f} seconds')
    return u, v, w

