from pacmof2 import pacmof2

path_to_cif = 'ddec'
output_path = 'pacmof'

# 1.Single CIF
path_to_cif = 'ddec/JAMLAL_with_charge.cif'
pacmof2.get_charges(path_to_cif, output_path, identifier="_pacmof", net_charge=4)

# 2.Multiple CIFs
net_charges = {"JAMLAL_with_charge.cif": 4.0, "UCIBAK_with_charge.cif": -4.0}
path_to_cif = 'ddec'
pacmof2.get_charges(path_to_cif, output_path, identifier='_pacmof', multiple_cifs=True, net_charge=net_charges)

# 3.Multiple CIFs from json
import json
path_to_cif = 'ddec'
with open('net_charges.json', 'r') as f:
    net_charges = json.load(f)
pacmof2.get_charges(path_to_cif, output_path, identifier='_pacmof', multiple_cifs=True, net_charge=net_charges)
