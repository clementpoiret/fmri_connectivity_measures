import getopt
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nilearn import datasets, plotting
from nilearn.connectome import ConnectivityMeasure
from nilearn.image import load_img
from nilearn.input_data import NiftiMapsMasker

#_path = './data/'
DEFAULT_KIND = 'correlation'
DEFAULT_FILTER = '**/*bandpassed*.nii.gz'


def load_atlas():
    """Loading a provided atlas
    
    Returns:
        {Nibabel Image, list} -- Atlas's path and related labels
    """
    atlas = datasets.fetch_atlas_msdl()
    atlas_filename = atlas['maps']
    labels = atlas['labels']

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

    correlation_measure = ConnectivityMeasure(kind='correlation')
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
    _kind = ''
    _filter = ''

    try:
        opts, args = getopt.getopt(argv, 'hp:k:f:',
                                   ['path=', 'kind=', 'filter='])
    except getopt.GetoptError:
        print('app.py -p <path> -k <kind> -f <filter>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('app.py -p <path> -k <kind> -f <filter>')
            sys.exit()
        elif opt in ('-p', '--path'):
            _path = arg
        elif opt in ('-k', '--kind'):
            _kind = arg
        elif opt in ('-f', '--filter'):
            _filter = arg

    if not _path:
        print(
            'Arguement \'-p <path>\' required. Please provide a path containing .nii.gz files.'
        )

    _kind = _kind or DEFAULT_KIND
    _filter = _filter or DEFAULT_FILTER

    p = Path(_path)
    fmris = p.glob(_filter)

    print('Loading atlas...')
    atlas_filename, labels = load_atlas()

    matrices = {}

    for fmri in fmris:
        fmri = fmri.as_posix()

        loaded_fmri = load_fmri(fmri)

        correlation_matrix = get_correlation_matrix(loaded_fmri,
                                                    atlas_filename,
                                                    standardize=True,
                                                    kind=_kind)

        matrices[fmri] = correlation_matrix

    with open(p / '{}_matrices.pkl'.format(_kind), 'wb') as handle:
        print('Saving {} matrices to \'{}\'...'.format(
            _kind, p / '{}_matrices.pkl'.format(_kind)))
        pickle.dump(matrices, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #plot(correlation_matrix, labels)

    if not matrices:
        print('No .nii.gz file found. Please provide change path or filter.')


if __name__ == "__main__":
    main(sys.argv[1:])
