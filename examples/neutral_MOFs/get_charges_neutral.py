from pacmof2 import pacmof2

path_to_cif = 'ddec'
output_path = 'pacmof'

# 1.Single CIF
pacmof2.get_charges("ddec/UIO-66.cif", output_path, identifier="_pacmof", adjust_charge_method="mean")

# 2.Multiple CIFs
pacmof2.get_charges(path_to_cif, output_path, identifier='_pacmof', multiple_cifs=True, adjust_charge_method='mean')

# 3.Print features
pacmof2.get_charges(path_to_cif, output_path, identifier='_pacmof', multiple_cifs=True, adjust_charge_method='mean', print_features=True)
