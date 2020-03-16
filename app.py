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
DEFAULT_KIND = 'correlation'
DEFAULT_FILTER = '**/*bandpassed*.nii.gz'
DEFAULT_DOWNLOAD_PATH = './downloaded_atlas.nii.gz'


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
    atlas_filename = ''
    labels = []

    if not atlas_location:
        atlas = datasets.fetch_atlas_msdl()
        atlas_filename = atlas['maps']
        labels = atlas['labels']

    else:
        if is_url(atlas_location):
            print('Beginning atlas download with urllib2...')
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


def get_correlation_matrix(image,
                           atlas_filename,
                           standardize=True,
                           kind='correlation'):
    """Computes correlation matrix on a given nibabel image
    
    Arguments:
        image {Nibabel Image} -- fMRI
        atlas_filename {str} -- Path to atlas
    
    Keyword Arguments:
        standardize {bool} -- Standardize (default: {True})
        kind {str} -- Nilearn's kind of Connectivity Measure (default: {'correlation'})
    
    Returns:
        {array} -- Correlation Matrix
    """
    masker = NiftiMapsMasker(maps_img=atlas_filename,
                             standardize=True,
                             verbose=5)

    time_series = masker.fit_transform(image)

    correlation_measure = ConnectivityMeasure(kind=kind)
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    z_correlation_matrix = np.arctanh(correlation_matrix)

    return z_correlation_matrix


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


def process_fmris(fmris, atlas, kind, subjects):
    matrices = {}

    subjects_list = pd.read_csv(subjects, header=None)
    for fmri in fmris:
        subject_id = [s for s in subjects_list[0] if s in str(fmri)]

        if not subject_id:
            print('Found fmri without corresponding ID. Skipping \'{}\''.format(
                str(fmri)))
            continue

        print('Processing {}'.format(subject_id[0]))
        fmri = fmri.as_posix()

        loaded_fmri = load_fmri(fmri)

        z_correlation_matrix = get_correlation_matrix(loaded_fmri,
                                                      atlas,
                                                      standardize=True,
                                                      kind=kind)

        matrices[subject_id[0]] = z_correlation_matrix

    return matrices


def save_matrices(matrices, path, n_subjects, kinds):
    _filename = '{}_{}subjects_{}.pkl'.format(
        time.time_ns() // 1000000, n_subjects,
        str([kind for kind in kinds
            ]).strip('[]').replace(', ', '_').replace("'",
                                                      '').replace(' ', '-'))

    with open(path / _filename, 'wb') as handle:
        print('Saving matrices to \'{}\'...'.format(path / _filename))
        pickle.dump(matrices, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main(args):
    _path = args.path
    _filter = args.filter
    atlas = args.atlas
    kinds = [kind for kind in args.kind.split(',')]
    subjects = args.subjects
    download_path = args.downloadpath

    print('Loading atlas...')
    atlas, labels = load_atlas(atlas, download_path=download_path)

    if not _path:
        print(
            'Argument \'-p <path>\' suggested. Please provide a path containing .nii.gz files. Checking current folder...'
        )

    p = Path(_path)
    fmris = sorted(p.glob(_filter))

    if not atlas:
        print('Loading atlas...')
        atlas, labels = load_atlas()

    matrices = {}
    for kind in kinds:
        print('Computing {}'.format(kind))
        matrices[kind] = process_fmris(fmris, atlas, kind, subjects)

    if not matrices:
        print('No .nii.gz file found. Please update path or filter.')
    else:
        save_matrices(matrices, p, len(matrices[kinds[0]]) + 1, kinds)

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
        'Comma separated list of nilearn\'s kinds (e.g.: "partial correlation,correlation,tangent"). Default: "correlation"',
        default=[DEFAULT_KIND],
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
