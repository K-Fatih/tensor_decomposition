import h5py
import numpy as np


# The code reads an h5 file and loads the stored strain modes
# EMMA data format xx,yy,zz,xy,xz,yz
# the file is expected to have the following fileds
# ['/material/number_of_phases','/material/material_index','/mesh/integration_coefficients','/reduced_basis/strain_modes']

def load_reduced_basis(file_name='data/laminate_sphere.h5', number_of_modes=-1):
    h5_file = h5py.File(file_name, 'r')

    number_of_phases = (h5_file['/material/number_of_phases'][:].astype(int)).item()
    material_index = h5_file['/material/material_index'][:].astype(int)
    integration_coefficients = h5_file['/mesh/integration_coefficients'][:]

    assert '/reduced_basis/strain_modes' in h5_file, 'This type of modes is not stored in the h5_file'
    strain_modes = h5_file['/reduced_basis/strain_modes'][:]

    n_modes = number_of_modes if number_of_modes in range(1, strain_modes.shape[-1]) else strain_modes.shape[-1]

    truncated_strain_modes = strain_modes[:, :, :n_modes]

    L = np.sqrt(np.loadtxt('{}_spectrum.txt'.format(file_name[:-3])))
    singular_values = L[:n_modes] / L[0]

    h5_file.close()

    print('done reading h5 file')

    return number_of_phases, material_index, integration_coefficients, truncated_strain_modes, singular_values

if __name__ == '__main__':
    number_of_phases, material_index, weights, truncated_strain_modes, singular_values = load_reduced_basis(file_name='data/laminate_sphere.h5', number_of_modes=64)

    print(weights.shape)
    print(truncated_strain_modes.shape)
    input_tensor = np.einsum('...k,k->...k', truncated_strain_modes, singular_values, optimize='optimal')
    print(input_tensor.shape)

    # n_load_cases = 10
    # val_data = h5py.File('data/laminate_sphere_verification.h5', 'r')
    # idx_range = np.arange(0, n_load_cases)
    # average_strains = val_data['training/strain'][idx_range, :]
    # average_stress = val_data['training/stress_ROM96'][idx_range, :]
    # val_data.close()
    # print(average_strains.shape)
    # print(average_stress.shape)
