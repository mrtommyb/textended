import numpy as np
import pandas as pd
import sys
# import astroquery
# import matplotlib.pyplot as plt
# import glob
from tqdm import tqdm
# import matplotlib
from tvguide import TessPointing
from astropy.coordinates import SkyCoord
from astropy import units as u
from numpy.random import poisson, beta, uniform
from numpy import array as nparr
import simfuncs

from make_catalog import *

tqdm.pandas()

consts = {'sigma_threshold': 10, 
          'detect_transits': 3,
          'sector_length': 13.7,
          'version': 'OST300',
         }


def run_sim(df, i):
    newDF = calculate_planet_properties(df)

    selected = newDF[newDF.has_transits == True]
    # selected.to_csv('../data/allCTL7-EM-{}-{}T.csv.bz2'.format(consts['version'], consts['detect_transits']),
    #           compression='bz2')

    out_SNE = get_camera_bouma(selected.reset_index(drop=True), fieldfile='../data/camera_boresights_SNE.csv')

    df_out_SNE = pd.DataFrame(out_SNE, columns=[str(x) for x in range(1, 1+out_SNE.shape[1])])

    dfw_SNE = pd.concat([selected.reset_index(drop=True), df_out_SNE], axis=1)
    nsectors = out_SNE.shape[1]
    dfw_SNE = make_output_arr(dfw_SNE, nsectors)
    print('Planets detected in primary + extended mission SNE: {}'.format(dfw_SNE[dfw_SNE.detected].shape[0]))
    print('Planets detected in primary mission SNE: {}'.format(dfw_SNE[dfw_SNE.detected_primary].shape[0]))
    dfw_SNE.to_csv('../data/OST_300/obs_SNE-{0}-{1}T-{2:03d}.csv.bz2'.format(consts['version'], consts['detect_transits'],
                                                                      i),
              compression='bz2')


if __name__ == '__main__':

    fn = '../data/exo_CTL_07.02xTIC_v7.csv'

    header = [
            'TICID', 'RA', 'DEC', 'PLX', 'ECLONG', 'ECLAT', 
            'V', 'J', 'Ks', 'TESSMAG', 'TEFF', 
            'RADIUS', 'MASS', 'CONTRATIO', 'PRIORITY'
            ]

    usecols = [0, 13, 14, 21, 26, 27, 30, 42, 46, 60, 64, 70, 72, 84, 87,]

    print()
    print('doing entire CTL')

    dfo = pd.read_csv(fn, names=header, usecols=usecols)

    for i in range(300):
        df = dfo.copy()
        run_sim(df, i)
