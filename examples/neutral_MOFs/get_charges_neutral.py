from pacmof2 import pacmof2

path_to_cif = 'ddec'
output_path = 'pacmof'

# 1.Single CIF
path_to_cif = 'ddec/LASYOU_clean_DDEC.cif'
pacmof2.get_charges(path_to_cif, output_path, identifier="_pacmof")

# 2.Multiple CIFs
path_to_cif = 'ddec'
pacmof2.get_charges(path_to_cif, output_path, identifier='_pacmof', multiple_cifs=True)

# 3.Print features
path_to_cif = 'ddec'
pacmof2.get_charges(path_to_cif, output_path, identifier='_pacmof', multiple_cifs=True, print_features=True)
