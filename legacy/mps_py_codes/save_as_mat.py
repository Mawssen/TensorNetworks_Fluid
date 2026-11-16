import scipy.io
import h5py
import numpy as np

def save_as_mat(file_name, result):
    """
    Save the result dictionary to a .mat file.

    Parameters
    ----------
    file_name : str
        The file name without the .mat extension.
    result : dict
        The dictionary of results to be saved.
    """
    mat_file_name = f'{file_name}.mat'
    scipy.io.savemat(mat_file_name, result)
    print(f'Saved result to {mat_file_name}')


def save_as_h5(file_name, var_name, data):
    """Save a large NumPy array as an HDF5 file."""
    with h5py.File(f"{file_name}.h5", "w") as f:
        f.create_dataset(var_name, data=data)
    print(f"Saved {file_name}.h5 successfully.")

# def save_as_mat_hdf5(file_name, data_dict):
#     with h5py.File(file_name + '.mat', 'w') as f:
#         for key, value in data_dict.items():
#             f.create_dataset(key, data=value)

#     print(f'Saved result to {file_name}.mat')

# Modify save_as_mat_hdf5 to handle inhomogeneous data

def save_as_mat_hdf5(file_name, data_dict):
    with h5py.File(file_name + '.mat', 'w') as f:
        for key, value in data_dict.items():
            # If value is a list or has multiple elements, save each one individually
            if isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    # Transpose the data before saving
                    f.create_dataset(f'{key}_{i}', data=np.transpose(v))
            else:
                # Transpose the data before saving
                f.create_dataset(key, data=np.transpose(value))

    print(f'Saved result to {file_name}.mat')


def load_h5(file_name, n=1):
    with h5py.File(file_name, "r") as h5file:
        dataset_name = list(h5file.keys())[n-1]
        data = h5file[dataset_name][()]   # type: ignore
    return data

