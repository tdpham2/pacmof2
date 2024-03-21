from pacmof2 import pacmof2
import glob

path_to_cif = 'example_ionic'
output_path = 'example_output'
multiple_cifs = False

net_charge = {
'AJACEW_with_charge.cif':-2,
'AWESIF_with_charge.cif':+2  ,
'UIO-66.cif': 0        
        }
pacmof2.get_charges(path_to_cif, output_path, identifier='_pacmof_ionic', multiple_cifs=True, adjust_charge_method='mean', net_charge=net_charge, print_features=True)
