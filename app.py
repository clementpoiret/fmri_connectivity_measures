import argparse
import pickle
import urllib.request
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import validators
from nilearn import datasets, plotting
from nilearn.connectome import ConnectivityMeasure
from nilearn.image import load_img
from nilearn.input_data import NiftiMapsMasker

#_path = './data/'
DEFAULT_KINDS = ['correlation', 'partial correlation', 'tangent']
DEFAULT_FILTER = '**/*bandpassed*.nii.gz'
DEFAULT_DOWNLOAD_PATH = './downloaded_atlas.nii.gz'


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def is_url(s):
    return False if not validators.url(s) else True


def load_atlas(atlas_location=None, download_path=DEFAULT_DOWNLOAD_PATH):
    """Loading a provided atlas
    
    Keyword Arguments:
        atlas_location {str} -- path or url to the atlas (default: {None})
        download_path {[type]} -- download path for the atlas(default: {'./downloaded_atlas.nii.gz'})
    
    Returns:
        {Nibabel Image, list} -- Atlas's path and related labels
    """
    print(f'{bcolors.OKBLUE}Loading atlas{bcolors.ENDC}')
    atlas_filename = ''
    labels = []

    if not atlas_location:
        atlas = datasets.fetch_atlas_msdl()
        atlas_filename = atlas['maps']
        labels = atlas['labels']

    else:
        if is_url(atlas_location):
            print(
                f'{bcolors.OKBLUE}Beginning atlas download with urllib2...{bcolors.ENDC}'
            )
            urllib.request.urlretrieve(atlas_location, download_path)

            atlas_filename = download_path
        else:
            atlas_filename = atlas_location

    return (atlas_filename, labels)


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
                        standardize=True,
                        verbose=5):
    """Extracting time series from a list of fmris
    
    Arguments:
        fmris {list} -- List of loaded fMRIs
        subjects_list {list} -- List of subjects' IDs
        atlas {str} -- Path to atlas
    
    Keyword Arguments:
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
        time_series = masker.fit_transform(img)

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

    for kind in kinds:
        if kind == 'tangent' and len(set(subjects)) < 2:
            print(
                f'{bcolors.FAIL}Tangent space parametrization can only be applied to a group of subjects, as it returns deviations to the mean. Skipping{bcolors.ENDC}'
            )
            continue

        print(f'{bcolors.OKBLUE}Computing {kind}{bcolors.ENDC}')

        connectivity_measures = ConnectivityMeasure(kind=kind)

        connectivity_matrices = connectivity_measures.fit_transform(time_series)

        matrices[kind] = {
            subjects[i]: connectivity_matrices[i]
            for i in range(connectivity_matrices.shape[0])
        }

    return matrices


def plot(correlation_matrix, labels):
    """Plot the correlation matrix
    
    Arguments:
        correlation_matrix {array} -- Correlation Matrix
        labels {list} -- Labels
    """
    np.fill_diagonal(correlation_matrix, 0)
    plotting.plot_matrix(correlation_matrix,
                         labels=labels,
                         colorbar=True,
                         vmax=0.8,
                         vmin=-0.8)
    plt.show()


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
    atlas = args.atlas
    kinds = [kind for kind in args.kind.split(',')]
    subjects = args.subjects
    download_path = args.downloadpath

    atlas, _ = load_atlas(atlas, download_path=download_path)

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

    subjects_list = pd.read_csv(subjects, header=None)[0]

    subjects_time_series, processed_subjects = extract_time_series(
        fmris=fmris,
        subjects_list=subjects_list,
        atlas=atlas,
        standardize=True,
        verbose=5)

    matrices = get_connectivity_matrices(
        time_series=subjects_time_series,
        subjects=processed_subjects,
        kinds=kinds,
    )

    save_matrices(matrices, p, len(matrices[kinds[0]]), kinds)

    #plot(correlation_matrix, labels)


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
        help='Regex filter to select fMRI. Default: "**/*bandpassed*.nii.gz"',
        default=DEFAULT_FILTER,
        required=False)
    parser.add_argument(
        '-s',
        '--subjects',
        help=
        '<Required> Path to a csv file with an "ID" column. Every fmri should have its ID in its relative path',
        required=True)

    args = parser.parse_args()

    main(args)
