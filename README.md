# Connectivity Measures

Required packages:

- Nibabel,
- Nilearn,
- Scikit Learn,
- validators.

usage: app.py [-h] [-p PATH] [-a ATLAS] [-r REGIONS] [-d DOWNLOADPATH]
              [-k KIND] [-f FILTER] -s SUBJECTS

Currently supported connectivity measures:

- correlation,
- partial correlation,
- tangent,
- covariance,
- precision.

Computes connectivity matrices of fmris.

```
arguments:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  Path to a folder containing fMRI
  -a ATLAS, --atlas ATLAS
                        URL or Local Path to an atlas
  -r REGIONS, --regions REGIONS
                        Local Path to a list of brain regions
  -d DOWNLOADPATH, --downloadpath DOWNLOADPATH
                        Path with filename for the downloaded atlas. Default:
                        ./downloaded_atlas.nii.gz
  -k KIND, --kind KIND  Comma separated list of nilearn's kinds (e.g.:
                        "partial correlation,correlation,tangent,covariance,pr
                        ecision"). Default: "correlation,partial correlation"
  -f FILTER, --filter FILTER
                        Regex filter to select fMRI. Default:
                        "**/*bandpassed*.nii.gz"
  -s SUBJECTS, --subjects SUBJECTS
                        <Required> Path to a csv file or a list of csv files,
                        with IDs in the first column. Every fmri should have
                        its ID in its relative path
```

Example:

```python app.py -p './data/BIDS/' -a 'https://my_url.com/msdl_rois.nii' -d './msdl_atlas.nii' -k 'partial correlation' -f '**/*bandpassed*.nii.gz' -s './subjects1.csv,subjects2.csv'```

The above command will compute a partial correlation matrix for each `.nii.gz` file containing
the word `bandpassed`, in the folder `./data/BIDS/`, including subfolders, using an atlas downloaded
from `https://my_url.com/msdl_rois.nii` to `./msdl_atlas.nii`.

Note that `-a` supports both local path and valid url.

**If you provide an URL, it is good practice to define a download path using `-d`, to ensure the online atlas and the downloaded one have the same extension (.e.g.: .nii, .nii.gz, etc.)**
Default download location: `./downloaded_atlas.nii.gz`.

When no atlas is specified, it defaults to MSDL.

Each path should contain an unique identifier (in a folder or file name) for a given subject. Identifiers should be reported in **the first column** of `./subjects1.csv` and `subjects2.csv` or any other specified csv file by `-s`. The script also supports list of csv files.

Matrices will be saved as values in a Dict, with IDs as keys, in a pickle file saved in `<path>/{time}_{n}subjects_[kinds].pkl`.
