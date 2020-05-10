""" 
    Computes a new atlas for a given list of regions.
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

import os
import time
import urllib.request

import nibabel as nib
import numpy as np
import pandas as pd

MIST_BASE_URL = 'https://github.com/clementpoiret/fmri_connectivity_measures/raw/master/files/MIST/'


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


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


def main():
    regions = pd.read_csv('example.csv').values
    locations = get_regions_locations(regions)
    autoatlas(locations)


if __name__ == '__main__':
    main()