from pacmof2 import pacmof2
import glob

path_to_cif = 'example_ionic'
output_path = 'example_output'
multiple_cifs = False

pacmof2.get_charges(path_to_cif, output_path, identifier='_pacmof_ionic', multiple_cifs=True, adjust_charge_method='mean', net_charge=0, print_features=True)
