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
import simfuncs

def get_camera(df, strategy='SNSNS'):
    # now add tvguide cameras
    year1cameras = np.zeros((df.shape[0], 13), dtype='int')
    year2cameras = np.zeros((df.shape[0], 13), dtype='int')
    year3cameras = np.zeros((df.shape[0], 13), dtype='int')
    year4cameras = np.zeros((df.shape[0], 13), dtype='int')
    year5cameras = np.zeros((df.shape[0], 13), dtype='int')

    if strategy == 'SNSNS':
        for i in tqdm(df.index):
            objS = TessPointing(df.loc[i, 'RA'], df.loc[i, 'DEC'])
            camerasS = objS.get_13cameras()
            year1cameras[i] = camerasS
            year3cameras[i] = camerasS
            year5cameras[i] = camerasS

            # hack to northern targets
            gc = SkyCoord(lon=df.loc[i, 'ECLONG'] * u.degree,
                          lat=df.loc[i, 'ECLAT'] * u.degree * -1,
                          frame='barycentrictrueecliptic')
            objN = TessPointing(gc.icrs.ra.value, gc.icrs.dec.value)
            camerasN = objN.get_13cameras()
            year2cameras[i] = camerasN
            year4cameras[i] = camerasN

    elif strategy == 'SNNSN':
        for i in tqdm(df.index):
            objS = TessPointing(df.loc[i, 'RA'], df.loc[i, 'DEC'])
            camerasS = objS.get_13cameras()
            year1cameras[i] = camerasS
            year4cameras[i] = camerasS

            # hack to northern targets
            gc = SkyCoord(lon=df.loc[i, 'ECLONG'] * u.degree,
                          lat=df.loc[i, 'ECLAT'] * u.degree * -1,
                          frame='barycentrictrueecliptic')
            objN = TessPointing(gc.icrs.ra.value, gc.icrs.dec.value)
            camerasN = objN.get_13cameras()
            year2cameras[i] = camerasN
            year3cameras[i] = camerasN
            year5cameras[i] = camerasN

    elif strategy == 'SNE':
        for i in tqdm(df.index):
            # for SNE the sector order is
            # year 1: south, 
            # year 2: north
            # year 3: NNNEEEEESSSSS
            # year 4: SSSSSSSSNNNNN
            # year 5: NNNNNNNNNNNNN
            objS = TessPointing(df.loc[i, 'RA'], df.loc[i, 'DEC'])
            gc = SkyCoord(lon=df.loc[i, 'ECLONG'] * u.degree,
                          lat=df.loc[i, 'ECLAT'] * u.degree * -1,
                          frame='barycentrictrueecliptic')
            objN = TessPointing(gc.icrs.ra.value, gc.icrs.dec.value)
            camerasS = objS.get_13cameras()
            camerasN = objN.get_13cameras()
            year1cameras[i] = camerasS
            year2cameras[i] = camerasN
            year5cameras[i] = camerasN

            year4cameras[i] = camerasN
            year4cameras[i, 0:8] = camerasS[0:8]

            year3cameras[i] = camerasS
            year3cameras[i, 0:3] = camerasN[0:3]

        ecl_pointings = get_ecl_pointings(df)
        year3cameras[:, 3:8] = ecl_pointings

    return [year1cameras, year2cameras, year3cameras, year4cameras, year5cameras]


def get_ecl_pointings(df, start=0, nsectors=5):
    #draw a rectangle of +/- 12 degrees, and 96 degrees
    outarr = np.zeros((df.shape[0], nsectors), dtype='int')
    for snum in range(nsectors):
        elon = df.loc[:, 'ECLONG']
        elat = df.loc[:, 'ECLAT']
        emin = start
        emax = start+96
        mask = ((elon >= emin) & (elon < emax) &
                np.abs(elat <= 12))
        outarr[mask,snum] = 9
        start += 27.7
    return outarr








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

    df = pd.read_csv(fn, names=header, usecols=usecols)
    df['isMdwarf'] = np.where((df.TEFF < 3900) & (df.RADIUS < 0.6), True, False)
    df['isGiant'] = np.ones(df.shape[0], dtype='bool') # assume all dwarfs
    df['isSubgiant'] = np.where(np.random.random(size=df.shape[0]) < 0.25, True, False)
    df.loc[df.isSubgiant & ~df.isMdwarf, 'RADIUS'] = df.RADIUS * 2
    df['cosi'] = pd.Series(np.random.random(size=df.shape[0]),name='cosi')
    df['noise_level'] = simfuncs.component_noise(df.TESSMAG, readmod=1, zodimod=1)

    # I need to change the lambdas to account for the increased parameter space size
    # for fgk going from 0.689 -> 1.10 because there are 60 percent more balls
    # for m going from 2.5 -> 2.96 because there are 18 percent more balls
    np_fgk = poisson(lam=1.10,size=df.shape[0])
    np_m = poisson(lam=2.96,size=df.shape[0])
    df['Nplanets'] = pd.Series(np.where(df.isMdwarf, np_m, np_fgk), name='Nplanets')

    starID = 0 # ???
    newDF, starID = simfuncs.make_allplanets_df_vec_extrap(df, starID)
    newDF = newDF.assign(T0=pd.Series(uniform(0, 1, size=newDF.shape[0]) * newDF.loc[:, 'planetPeriod']))

    newDF['ars'] = simfuncs.per2ars(newDF.planetPeriod, newDF.MASS, newDF.RADIUS)
    newDF['ecc'] = pd.Series(beta(1.03,13.6,size=newDF.shape[0]),name='ecc', ) # ecc dist from Van Eylen 2015
    newDF['omega'] = pd.Series(uniform(-np.pi,np.pi,size=newDF.shape[0]),name='omega')
    newDF['rprs'] = simfuncs.get_rprs(newDF.planetRadius, newDF.RADIUS)
    newDF['impact'] = newDF.cosi * newDF.ars * ((1-newDF.ecc**2)/1+newDF.ecc*np.sin(newDF.omega)) # cite Winn
    newDF['duration'] = simfuncs.get_duration(newDF.planetPeriod, newDF.ars, cosi=newDF.cosi, b=newDF.impact,
                                    rprs=newDF.rprs) # cite Winn
    newDF['duration_correction'] = np.sqrt(newDF.duration * 24.) # correction for CDPP because transit dur != 1 hour
    newDF['transit_depth']  = simfuncs.get_transit_depth(newDF.planetRadius, newDF.RADIUS)

    newDF['transit_depth_diluted']  = newDF['transit_depth'] / (1+newDF.CONTRATIO)
    newDF['has_transits']  = (newDF.ars > 1.0) & (newDF.impact < 1.0)


    selected = newDF[newDF.has_transits == True]
    selected.to_csv('../data/allCTL7-EM-v2.csv.bz2',
                  compression='bz2')

    newDF.to_csv('/home/tom/Dropbox/filetransfer/allCTL7-EM-v2-everything.csv.bz2',
                  compression='bz2')
    
    
    out_SNE = get_camera(selected.reset_index(drop=True), strategy='SNE')
    out_SNSNS = get_camera(selected.reset_index(drop=True), strategy='SNSNS')
    out_SNNSN = get_camera(selected.reset_index(drop=True), strategy='SNNSN')

    df_out_SNE = pd.concat([
              pd.DataFrame(out_SNE[0], columns=[str(x) for x in range(1, 14)]), 
              pd.DataFrame(out_SNE[1], columns=[str(x) for x in range(14, 27)]), 
              pd.DataFrame(out_SNE[2], columns=[str(x) for x in range(27, 40)]), 
              pd.DataFrame(out_SNE[3], columns=[str(x) for x in range(40, 53)]), 
              pd.DataFrame(out_SNE[4], columns=[str(x) for x in range(53, 66)]), 
              ], axis=1)

    df_out_SNSNS = pd.concat([
              pd.DataFrame(out_SNSNS[0], columns=[str(x) for x in range(1, 14)]), 
              pd.DataFrame(out_SNSNS[1], columns=[str(x) for x in range(14, 27)]), 
              pd.DataFrame(out_SNSNS[2], columns=[str(x) for x in range(27, 40)]), 
              pd.DataFrame(out_SNSNS[3], columns=[str(x) for x in range(40, 53)]), 
              pd.DataFrame(out_SNSNS[4], columns=[str(x) for x in range(53, 66)]), 
              ], axis=1)

    df_out_SNNSN = pd.concat([
              pd.DataFrame(out_SNNSN[0], columns=[str(x) for x in range(1, 14)]), 
              pd.DataFrame(out_SNNSN[1], columns=[str(x) for x in range(14, 27)]), 
              pd.DataFrame(out_SNNSN[2], columns=[str(x) for x in range(27, 40)]), 
              pd.DataFrame(out_SNNSN[3], columns=[str(x) for x in range(40, 53)]), 
              pd.DataFrame(out_SNNSN[4], columns=[str(x) for x in range(53, 66)]), 
              ], axis=1)

    df_out_SNE.to_csv('../data/obs_SNE-v2.csv.bz2',
                  compression='bz2')
    df_out_SNSNS.to_csv('../data/obs_SNSNS-v2.csv.bz2',
                  compression='bz2')
    df_out_SNNSN.to_csv('../data/obs_SNNSN-v2.csv.bz2',
                  compression='bz2')