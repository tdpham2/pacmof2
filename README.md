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
To be updated soon.

## Reference
To be updated soon.
