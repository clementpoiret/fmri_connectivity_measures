""" 
    Computes connectivity matrices of fmris for a given list of brain
    regions.
    Copyright (C) 2020 Clément POIRET

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import os
import pickle
import time
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import ppscore as pps
import validators
from nilearn import datasets, plotting
from nilearn.connectome import ConnectivityMeasure
from nilearn.image import load_img
from nilearn.input_data import NiftiMapsMasker
from sklearn.covariance import GraphicalLassoCV

DEFAULT_KINDS = ['correlation', 'partial correlation', 'tangent']
DEFAULT_FILTER = '**/*.nii.gz'
DEFAULT_CONFOUNDS_FILTER = '**/*reducedConfound*.tsv'
DEFAULT_DOWNLOAD_PATH = './downloaded_atlas.nii.gz'
REGIONS_URL = 'https://raw.githubusercontent.com/clementpoiret/fmri_connectivity_measures/master/files/regions.csv'
MIST_BASE_URL = 'https://github.com/clementpoiret/fmri_connectivity_measures/raw/master/files/MIST/'
NILEARN_KINDS = ['correlation', 'partial correlation', 'tangent']
SKLEARN_KINDS = ['covariance', 'precision']


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def download_regions():
    """Download regions location
    """
    if not os.path.exists('files'):
        os.mkdir('files')

    print(f'{bcolors.OKBLUE}Updating regions{bcolors.ENDC}')
    urllib.request.urlretrieve(REGIONS_URL, 'files/regions.csv')


def download_mist(resolutions):
    """Download specific resolutions of the MIST atlas
    
    Arguments:
        resolutions {list} -- List of resolutions
    """
    if not os.path.exists('files/MIST/'):
        os.makedirs('files/MIST/')

    for resolution in resolutions:
        if not os.path.exists(f'files/MIST/{resolution}.nii.gz'):
            url = MIST_BASE_URL + resolution + '.nii.gz'
            print(
                f'{bcolors.OKBLUE}Downloading {resolution} from {url}{bcolors.ENDC}'
            )
            urllib.request.urlretrieve(url, f'files/MIST/{resolution}.nii.gz')
        else:
            print(
                f'{bcolors.OKBLUE}{resolution} already downloaded{bcolors.ENDC}'
            )


def get_regions_locations(regions):
    """Get regions locations from name
    
    Arguments:
        regions {list} -- list of MIST's regions
    
    Returns:
        {list} -- Regions' locations and relative atlases
    """
    regions_locator = pd.read_csv('files/regions.csv', index_col='name')

    locations = np.array(
        [regions_locator.loc[region].values[0] for region in regions])
    print(
        f'{bcolors.OKBLUE}Got {len(locations)} regions from {len(set(locations[:, 1]))} atlases{bcolors.ENDC}'
    )

    return locations


def autoatlas(locations):
    """Computes a new atlas
    
    Arguments:
        locations {array} -- Output of get_regions_locations()
    
    Returns:
        {str} -- Path to the new atlas
    """
    required_atlases = set(locations[:, 1])
    download_mist(required_atlases)

    regions = []
    for value, resolution, _, _, _ in locations:
        mist_path = f'files/MIST/{resolution}.nii.gz'
        mist = nib.load(mist_path)
        region = (mist.get_fdata() == value) * 1

        region_img = nib.Nifti1Image(region, affine=mist.affine)
        regions.append(region_img)

    atlas = nib.concat_images(regions)
    filename = '{}_autoatlas_{}regions.nii'.format(time.time_ns() // 1000000,
                                                   atlas.shape[-1])
    nib.save(atlas, filename)

    print(
        f'{bcolors.OKGREEN}Success! Autoatlas saved to {filename}{bcolors.ENDC}'
    )

    return filename


def is_url(s):
    return False if not validators.url(s) else True


def load_atlas(atlas_location=None, download_path=DEFAULT_DOWNLOAD_PATH):
    """Loading a provided atlas
    
    Keyword Arguments:
        atlas_location {str} -- path or url to the atlas (default: {None})
        download_path {[type]} -- download path for the atlas(default: {'./downloaded_atlas.nii.gz'})
    
    Returns:
        {Nibabel Image} -- Atlas's path
    """
    print(f'{bcolors.OKBLUE}Loading atlas{bcolors.ENDC}')
    atlas_filename = ''

    if not atlas_location:
        atlas = datasets.fetch_atlas_msdl()
        atlas_filename = atlas['maps']

    else:
        if is_url(atlas_location):
            print(
                f'{bcolors.OKBLUE}Beginning atlas download with urllib2...{bcolors.ENDC}'
            )
            urllib.request.urlretrieve(atlas_location, download_path)

            atlas_filename = download_path
        else:
            atlas_filename = atlas_location

    return atlas_filename


def load_fmri(path):
    """Load fmri from a given path
    
    Arguments:
        path {str} -- Path to fmri
    
    Returns:
        {Nibabel Image}
    """
    image = load_img(path)

    return image


def extract_time_series(fmris,
                        subjects_list,
                        atlas,
                        confounds=None,
                        standardize=True,
                        verbose=5):
    """Extracting time series from a list of fmris
    
    Arguments:
        fmris {list} -- List of loaded fMRIs
        subjects_list {list} -- List of subjects' IDs
        atlas {str} -- Path to atlas
    
    Keyword Arguments:
        confounds {list<String>} -- List of confound's path (default: {None})
        standardize {bool} -- Standardize time series (default: {True})
        verbose {int} -- Verbosity (default: {5})
    
    Returns:
        {tuple} -- Returns time series and processed subjects's IDs
    """
    subjects_time_series = []
    processed_subjects = []

    for fmri in fmris:
        subject_id = [s for s in subjects_list if s in str(fmri)]

        if not subject_id:
            print(
                f'{bcolors.WARNING}Found fmri without corresponding ID. Skipping \'{str(fmri)}\'{bcolors.ENDC}'
            )
            continue

        print(f'{bcolors.OKBLUE}Loading {subject_id[0]}{bcolors.ENDC}')
        processed_subjects.append(subject_id[0])

        img = load_img(fmri.as_posix())
        masker = NiftiMapsMasker(maps_img=atlas,
                                 standardize=standardize,
                                 verbose=verbose)
        time_series = masker.fit_transform(img, confounds=confounds)

        subjects_time_series.append(time_series)

    return (subjects_time_series, processed_subjects)


def get_connectivity_matrices(time_series, subjects, kinds=DEFAULT_KINDS):
    """Computes connectivity matrices
    
    Arguments:
        time_series {list} -- List of extracted time series
        subjects {list} -- List of corresponding subbjects
    
    Keyword Arguments:
        kinds {list} -- List of connectivity measures (default: {DEFAULT_KINDS})
    
    Returns:
        {Numpy array} -- Connectivity matrices
    """
    matrices = {}
    n_subjects = len(set(subjects))

    for kind in kinds:
        print(
            f'{bcolors.OKBLUE}Computing {kind} of {n_subjects} subjects{bcolors.ENDC}'
        )

        if kind in NILEARN_KINDS:
            if kind == 'tangent' and n_subjects < 2:
                print(
                    f'{bcolors.FAIL}Tangent space parametrization can only be applied to a group of subjects, as it returns deviations to the mean. Skipping{bcolors.ENDC}'
                )
                continue

            connectivity_measures = ConnectivityMeasure(kind=kind)

            connectivity_matrices = connectivity_measures.fit_transform(
                time_series)

            matrices[kind] = {
                subjects[i]: connectivity_matrices[i]
                for i in range(connectivity_matrices.shape[0])
            }

        if kind in SKLEARN_KINDS:

            for i, subject in enumerate(subjects):
                estimator = GraphicalLassoCV()
                estimator = estimator.fit(time_series[i])

                matrix = None
                if kind == 'covariance':
                    matrix = estimator.covariance_

                if kind == 'precision':
                    matrix = estimator.precision_

                if not kind in matrices:
                    matrices[kind] = {}

                matrices[kind][subject] = matrix

        if kind == 'pps':
            for i, subject in enumerate(subjects):
                ts = pd.DataFrame(time_series[i])
                matrix = pps.matrix(ts, task='regression')
                np.fill_diagonal(matrix, 1)

                if not kind in matrices:
                    matrices[kind] = {}

                matrices[kind][subject] = matrix.values

    return matrices


def save_matrices(matrices, path, n_subjects, kinds):
    """[summary]
    
    Arguments:
        matrices {[type]} -- [description]
        path {[type]} -- [description]
        n_subjects {[type]} -- [description]
        kinds {[type]} -- [description]
    """
    _filename = '{}_{}subjects_{}.pkl'.format(
        time.time_ns() // 1000000, n_subjects,
        str([kind for kind in kinds
            ]).strip('[]').replace(', ', '_').replace("'",
                                                      '').replace(' ', '-'))

    with open(path / _filename, 'wb') as handle:
        print(
            f'{bcolors.OKBLUE}Saving matrices to \'{path / _filename}\'{bcolors.ENDC}'
        )
        pickle.dump(matrices, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'{bcolors.OKGREEN}Success!{bcolors.ENDC}')


def main(args):
    _path = args.path
    _filter = args.filter
    _confounds_filter = args.confoundsfilter
    _confounds_path = args.confoundspath
    atlas = args.atlas
    regions = args.regions
    kinds = [kind for kind in args.kind.split(',')]
    subjects_csvs = [csv for csv in args.subjects.split(',')]
    download_path = args.downloadpath

    if atlas and regions:
        raise ValueError(
            f'{bcolors.FAIL}You provided both atlas and regions. Please provide only one of them.{bcolors.ENDC}'
        )

    if regions:
        download_regions()

        regions = pd.read_csv(regions, header=None).to_numpy()

        locations = get_regions_locations(regions)

        atlas = autoatlas(locations)
    else:
        atlas = load_atlas(atlas, download_path=download_path)

    if not _path:
        print(
            f'{bcolors.WARNING}Argument \'-p <path>\' suggested. Please provide a path containing .nii.gz files. Checking current folder{bcolors.ENDC}'
        )

    p = Path(_path)
    fmris = sorted(p.glob(_filter))

    if not fmris:
        raise ValueError(
            f'{bcolors.FAIL}No .nii.gz file found. Please provide a valid path.{bcolors.ENDC}'
        )

    subjects_list = pd.concat(
        (pd.read_csv(f) for f in subjects_csvs)).iloc[:,
                                                      0].reset_index(drop=True)

    confounds = None
    if (_confounds_path):
        p = Path(_confounds_path)
        confounds = sorted(p.glob(_confounds_filter))
        confounds = [c.as_posix() for c in confounds]

    subjects_time_series, processed_subjects = extract_time_series(
        fmris=fmris,
        subjects_list=subjects_list,
        atlas=atlas,
        confounds=confounds,
        standardize=True,
        verbose=5)

    matrices = get_connectivity_matrices(
        time_series=subjects_time_series,
        subjects=processed_subjects,
        kinds=kinds,
    )

    save_matrices(matrices, p, len(matrices[kinds[0]]), kinds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Computes connectivity matrices of fmris.')

    parser.add_argument('-p',
                        '--path',
                        help='Path to a folder containing fMRI',
                        default='',
                        required=False)
    parser.add_argument('-a',
                        '--atlas',
                        help='URL or Local Path to an atlas',
                        required=False)
    parser.add_argument('-r',
                        '--regions',
                        help='Local Path to a list of brain regions',
                        required=False)
    parser.add_argument(
        '-d',
        '--downloadpath',
        help='Path with filename for the downloaded atlas. Default: {}'.format(
            DEFAULT_DOWNLOAD_PATH),
        default=DEFAULT_DOWNLOAD_PATH,
        required=False)
    parser.add_argument(
        '-k',
        '--kind',
        help=
        'Comma separated list of nilearn\'s kinds (e.g.: "partial correlation,correlation,tangent,covariance,precision"). Default: "correlation,partial correlation"',
        default=DEFAULT_KINDS,
        type=str)
    parser.add_argument(
        '-f',
        '--filter',
        help='Regex filter to select fMRI. Default: "**/*.nii.gz"',
        default=DEFAULT_FILTER,
        required=False)
    parser.add_argument('-c',
                        '--confoundspath',
                        help='Path to a folder containing fMRI\'s confounds',
                        default=None,
                        required=False)
    parser.add_argument(
        '-o',
        '--confoundsfilter',
        help=
        'Regex filter to select fMRI\'confounds. Default: "**/*reducedConfound*.tsv"',
        default=DEFAULT_CONFOUNDS_FILTER,
        required=False)
    parser.add_argument(
        '-s',
        '--subjects',
        help=
        '<Required> Path to a csv file or a list of csv files, with IDs in the first column. Every fmri should have its ID in its relative path',
        required=True,
        type=str)

    args = parser.parse_args()

    print("""Copyright (C) 2020  Clément POIRET
    This program comes with ABSOLUTELY NO WARRANTY; for help, launch it with `-h`.
    This is free software, and you are welcome to redistribute it
    under certain conditions.""")

    main(args)
