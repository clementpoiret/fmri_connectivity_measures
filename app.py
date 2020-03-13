import getopt
import pickle
import sys
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

    return correlation_matrix


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


def main(argv):
    _path = ''
    _atlas = ''
    _kind = ''
    _filter = ''
    download_path = DEFAULT_DOWNLOAD_PATH

    try:
        opts, args = getopt.getopt(
            argv, 'hp:a:d:k:f:',
            ['path=', 'atlas=', 'downloadpath='
             'kind=', 'filter='])
    except getopt.GetoptError:
        print(
            'app.py -p <path> -a <atlas> -d <downloadpath> -k <kind> -f <filter>'
        )
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(
                'app.py -p <path> -a <atlas> -d <downloadpath> -k <kind> -f <filter>'
            )
            sys.exit()
        elif opt in ('-p', '--path'):
            _path = arg
        elif opt in ('-d', '--downloadpath'):
            download_path = arg
        elif opt in ('-a', '--atlas'):
            print('Loading atlas...')
            _atlas, labels = load_atlas(arg, download_path=download_path)
        elif opt in ('-k', '--kind'):
            _kind = arg
        elif opt in ('-f', '--filter'):
            _filter = arg

    if not _path:
        print(
            'Arguement \'-p <path>\' required. Please provide a path containing .nii.gz files. Will check current folder.'
        )

    _kind = _kind or DEFAULT_KIND
    _filter = _filter or DEFAULT_FILTER

    p = Path(_path)
    fmris = p.glob(_filter)

    if not _atlas:
        print('Loading atlas...')
        _atlas, labels = load_atlas()

    matrices = {}

    for fmri in fmris:
        fmri = fmri.as_posix()

        loaded_fmri = load_fmri(fmri)

        correlation_matrix = get_correlation_matrix(loaded_fmri,
                                                    _atlas,
                                                    standardize=True,
                                                    kind=_kind)

        matrices[fmri] = correlation_matrix

    with open(p / '{}_matrices.pkl'.format(_kind), 'wb') as handle:
        print('Saving {} matrices to \'{}\'...'.format(
            _kind, p / '{}_matrices.pkl'.format(_kind)))
        pickle.dump(matrices, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #plot(correlation_matrix, labels)

    if not matrices:
        print('No .nii.gz file found. Please update path or filter.')


if __name__ == "__main__":
    main(sys.argv[1:])
