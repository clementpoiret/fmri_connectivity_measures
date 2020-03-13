# Connectivity Measures

Required packages:

- Nibabel,
- Nilearn.

To launch the script:

```python app.py -p <path> -a <atlas> -d <downloadpath> -k <kind> -f <filter>```

Example:

```python app.py -p './data/BIDS/' -a 'https://my_url.com/msdl_rois.nii' -d './msdl_atlas.nii' -k 'partial_correlation' -f '**/*bandpassed*.nii.gz'```

The above command will compute a partial correlation matrix for each `.nii.gz` file containing
the word `bandpassed`, in the folder `./data/BIDS/`, including subfolders, using an atlas downloaded
from `https://my_url.com/msdl_rois.nii` to `./msdl_atlas.nii`

Note that `-a` supports both local path and valid url.

**If you provide an URL, it is good practice to define a download path using `-d`, to ensure the online atlas and the downloaded one have the same extension (.e.g.: .nii, .nii.gz, etc.)**
Default download location: `./downloaded_atlas.nii.gz`.

When no atlas is specified, it defaults to MSDL.
