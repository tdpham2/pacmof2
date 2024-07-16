# Predicting Partial Atomic Charges in Metal-Organic Frameworks: An Extension to Ionic MOFs

Our publication is under review (Update soon).

## Overview
PACMOF2 is a Python package developed to predict partial atomic charges in Metal-Organic Frameworks (MOFs) with Density Functional Theory (DFT) level accuracy. PACMOF2 consists of two pre-trained machine learning models: `PACMOF2_neutral` and `PACMOF2_ionic`, aimed at predicting charges in both neutral and ionic MOFs. For more details on the method and implementation, please refer to our upcoming publication.

## Installation
PACMOF2 was thoroughly tested using Python 3.9. It requires the following dependencies:
- Pymatgen
- Atomic Simulation Environment (ASE)
- Scikit-Learn

To install PACMOF2, we recommend using an Anaconda environment:

```bash
conda create -n pacmof2 python==3.9
conda activate pacmof2
conda install -c conda-forge pymatgen
conda install -c conda-forge ase
conda install -c conda-forge scikit-learn=1.3.2
pip install build
```

### Downloading the Models
Due to file size limitations on GitHub, the PACMOF2 models are stored on Zenodo. Download the two models (`PACMOF2_ionic.gz` and `PACMOF2_neutral.gz`) directly from the Zenodo repository:

- [PACMOF2_ionic.gz](https://zenodo.org/records/12747095/files/PACMOF2_ionic.gz)
- [PACMOF2_neutral.gz](https://zenodo.org/records/12747095/files/PACMOF2_neutral.gz)

Place the downloaded models in the `pacmof2/models/` directory.

Alternatively, you can use the `wget` command to download the models:

```bash
wget -P pacmof2/models/ https://zenodo.org/records/12747095/files/PACMOF2_ionic.gz
wget -P pacmof2/models/ https://zenodo.org/records/12747095/files/PACMOF2_neutral.gz
```

### Installing PACMOF2
After installing the dependencies and downloading the models, install PACMOF2 using the following commands:

```bash
python3 -m build
pip install .
```

## Usage
PACMOF2 can be used to predict partial atomic charges in neutral MOFs (MOFs with a neutral unit cell) or ionic MOFs (MOFs with charged unit cells). Examples to run PACMOF2 are included in example/ folder. In general, to predict the charges of a neutral MOFs, run the following command:

from pacmof2 import pacmof2
path_to_cif = 'path/to/cif'
output_path = 'pacmof'
pacmof2.get_charges(path_to_cif, output_path, identifier="_pacmof")

PACMOF2 can also be used to predict the partial charges for a folder of CIFs. Here is an example:
from pacmof2 import pacmof2
path_to_cif = 'path/to/cifs'
output_path = 'pacmof'
pacmof2.get_charges(path_to_cifs, output_path, identifier='_pacmof', multiple_cifs=True)

For ionic MOFs, we need to specify the net charge(s) of the framework(s). For a single CIF, you can predict the charges using the following command:
from pacmof2 import pacmof2
path_to_cif = 'path/to/cif'
output_path = 'pacmof'


## Reference
To be updated soon.
