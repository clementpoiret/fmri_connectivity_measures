# Connectivity Measures

Required packages:

- Nibabel,
- Nilearn.

To launch the script:
```python app.py -p <path> -k <kind> -f <filter>```

Example:
```python app.py -p './data/BIDS/' -k 'partial_correlation' -f '**/*bandpassed*.nii.gz'```
The above command will compute a partial correlation matrix for each `.nii.gz` file containing
the word `bandpassed`, in the folder `./data/BIDS/`, including subfolders.
