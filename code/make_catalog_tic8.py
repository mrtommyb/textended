import numpy as np
import pandas as pd
import sys
# import astroquery
# import matplotlib.pyplot as plt
# import glob
from tqdm import tqdm
# import matplotlib
# from tvguide import TessPointing
from astropy.coordinates import SkyCoord
from astropy import units as u
from numpy.random import poisson, beta, uniform
from numpy import array as nparr
import simfuncs

tqdm.pandas()

# see https://github.com/lgbouma/tessmaps and https://github.com/lgbouma/extend_tess
from tessmaps.get_time_on_silicon import given_cameras_get_stars_on_silicon as gcgss

consts = {'sigma_threshold': 10,
          'detect_transits': 3,
          'sector_length': 13.7,
          'version': 'v5',
          }


# def get_camera(df, strategy='SNSNS'):
#     # now add tvguide cameras
#     year1cameras = np.zeros((df.shape[0], 13), dtype='int')
#     year2cameras = np.zeros((df.shape[0], 13), dtype='int')
#     year3cameras = np.zeros((df.shape[0], 13), dtype='int')
#     year4cameras = np.zeros((df.shape[0], 13), dtype='int')
#     year5cameras = np.zeros((df.shape[0], 13), dtype='int')

#     if strategy == 'SNSNS':
#         for i in tqdm(df.index):
#             objS = TessPointing(df.loc[i, 'RA'], df.loc[i, 'DEC'])
#             camerasS = objS.get_13cameras()
#             year1cameras[i] = camerasS
#             year3cameras[i] = camerasS
#             year5cameras[i] = camerasS

#             # hack to northern targets
#             gc = SkyCoord(lon=df.loc[i, 'ECLONG'] * u.degree,
#                           lat=df.loc[i, 'ECLAT'] * u.degree * -1,
#                           frame='barycentrictrueecliptic')
#             objN = TessPointing(gc.icrs.ra.value, gc.icrs.dec.value)
#             camerasN = objN.get_13cameras()
#             year2cameras[i] = camerasN
#             year4cameras[i] = camerasN

#     elif strategy == 'SNNSN':
#         for i in tqdm(df.index):
#             objS = TessPointing(df.loc[i, 'RA'], df.loc[i, 'DEC'])
#             camerasS = objS.get_13cameras()
#             year1cameras[i] = camerasS
#             year4cameras[i] = camerasS

#             # hack to northern targets
#             gc = SkyCoord(lon=df.loc[i, 'ECLONG'] * u.degree,
#                           lat=df.loc[i, 'ECLAT'] * u.degree * -1,
#                           frame='barycentrictrueecliptic')
#             objN = TessPointing(gc.icrs.ra.value, gc.icrs.dec.value)
#             camerasN = objN.get_13cameras()
#             year2cameras[i] = camerasN
#             year3cameras[i] = camerasN
#             year5cameras[i] = camerasN

#     elif strategy == 'SNE':
#         for i in tqdm(df.index):
#             # for SNE the sector order is
#             # year 1: south,
#             # year 2: north
#             # year 3: NNNEEEEESSSSS
#             # year 4: SSSSSSSSNNNNN
#             # year 5: NNNNNNNNNNNNN
#             objS = TessPointing(df.loc[i, 'RA'], df.loc[i, 'DEC'])
#             gc = SkyCoord(lon=df.loc[i, 'ECLONG'] * u.degree,
#                           lat=df.loc[i, 'ECLAT'] * u.degree * -1,
#                           frame='barycentrictrueecliptic')
#             objN = TessPointing(gc.icrs.ra.value, gc.icrs.dec.value)
#             camerasS = objS.get_13cameras()
#             camerasN = objN.get_13cameras()
#             year1cameras[i] = camerasS
#             year2cameras[i] = camerasN
#             year5cameras[i] = camerasN

#             year4cameras[i] = camerasN
#             year4cameras[i, 0:8] = camerasS[0:8]

#             year3cameras[i] = camerasS
#             year3cameras[i, 0:3] = camerasN[0:3]

#         ecl_pointings = get_ecl_pointings(df)
#         year3cameras[:, 3:8] = ecl_pointings

#     return [year1cameras, year2cameras, year3cameras, year4cameras, year5cameras]


def get_camera_bouma(df, fieldfile='../data/camera_boresights_SNE-shifted.csv'):
    """
    Like get_camera, only better
    """
    camdf = get_camera_coords(fieldfile)
    observed = np.zeros((df.shape[0], camdf.shape[0]), dtype='int')

    for ix, row in tqdm(camdf.iterrows(), total=camdf.shape[0]):
        cam_direction = row['camdirection']
        gc = SkyCoord(ra=df.loc[:, 'RA'].values * u.degree,
                      dec=df.loc[:, 'DEC'].values * u.degree,
                      frame='icrs')
        onchip = gcgss(gc, cam_direction, verbose=False)
        observed[:, ix] = onchip

    return observed


def get_camera_coords(camfn):
    camdf = pd.read_csv(camfn, sep=';')
    lats = nparr([
        nparr(camdf['cam1_elat']),
        nparr(camdf['cam2_elat']),
        nparr(camdf['cam3_elat']),
        nparr(camdf['cam4_elat'])]).T
    lons = nparr([
        nparr(camdf['cam1_elon']),
        nparr(camdf['cam2_elon']),
        nparr(camdf['cam3_elon']),
        nparr(camdf['cam4_elon'])]).T
    cam_directions = []
    for lat, lon in zip(lats, lons):

        c1lat, c2lat, c3lat, c4lat = lat[0], lat[1], lat[2], lat[3]
        c1lon, c2lon, c3lon, c4lon = lon[0], lon[1], lon[2], lon[3]

        this_cam_dirn = [(c1lat, c1lon),
                         (c2lat, c2lon),
                         (c3lat, c3lon),
                         (c4lat, c4lon)]

        cam_directions.append(this_cam_dirn)
    camdf['camdirection'] = cam_directions
    return camdf


def get_ecl_pointings(df, start=0, nsectors=5):
    # draw a rectangle of +/- 12 degrees, and 96 degrees
    outarr = np.zeros((df.shape[0], nsectors), dtype='int')
    for snum in range(nsectors):
        elon = df.loc[:, 'ECLONG']
        elat = df.loc[:, 'ECLAT']
        emin = start
        emax = start + 96
        mask = ((elon >= emin) & (elon < emax) &
                np.abs(elat <= 12))
        outarr[mask, snum] = 9
        start += 27.7
    return outarr


def calculate_planet_properties(df):
    df['isMdwarf'] = np.where(
        (df.TEFF < 3900) & (df.RADIUS < 0.6), True, False)
    df['isGiant'] = np.ones(df.shape[0], dtype='bool')  # assume all dwarfs
    # df['isSubgiant'] = np.where(np.random.random(
    #     size=df.shape[0]) < 0.25, True, False)
    # df.loc[df.isSubgiant & ~df.isMdwarf, 'RADIUS'] = df.RADIUS * 2
    df['cosi'] = pd.Series(np.random.random(size=df.shape[0]), name='cosi')
    df['noise_level'] = simfuncs.component_noise(
        df.TESSMAG, readmod=1, zodimod=1)

    # I need to change the lambdas to account for the increased parameter space size
    # for fgk going from 0.689 -> 1.10 because there are 60 percent more balls
    # for m going from 2.5 -> 2.96 because there are 18 percent more balls
    np_fgk = poisson(lam=1.10, size=df.shape[0])
    np_m = poisson(lam=2.96, size=df.shape[0])
    df['Nplanets'] = pd.Series(
        np.where(df.isMdwarf, np_m, np_fgk), name='Nplanets')

    starID = 0  # ???
    newDF, starID = simfuncs.make_allplanets_df_vec_extrap(df, starID)
    newDF = newDF.assign(T0=pd.Series(
        uniform(0, 1, size=newDF.shape[0]) * newDF.loc[:, 'planetPeriod']))

    newDF['ars'] = simfuncs.per2ars(
        newDF.planetPeriod, newDF.MASS, newDF.RADIUS)
    # ecc dist from Van Eylen 2015
    newDF['ecc'] = pd.Series(
        beta(1.03, 13.6, size=newDF.shape[0]), name='ecc', )
    newDF['omega'] = pd.Series(
        uniform(-np.pi, np.pi, size=newDF.shape[0]), name='omega')
    newDF['rprs'] = simfuncs.get_rprs(newDF.planetRadius, newDF.RADIUS)
    newDF['impact'] = newDF.cosi * newDF.ars * \
        ((1 - newDF.ecc**2) / 1 + newDF.ecc * np.sin(newDF.omega))  # cite Winn
    newDF['duration'] = simfuncs.get_duration(newDF.planetPeriod, newDF.ars, cosi=newDF.cosi, b=newDF.impact,
                                              rprs=newDF.rprs)  # cite Winn
    # correction for CDPP because transit dur != 1 hour
    newDF['duration_correction'] = np.sqrt(newDF.duration * 24.)
    newDF['transit_depth'] = simfuncs.get_transit_depth(
        newDF.planetRadius, newDF.RADIUS)

    newDF['transit_depth_diluted'] = newDF['transit_depth'] / \
        (1 + newDF.CONTRATIO)
    newDF['has_transits'] = (newDF.ars > 1.0) & (newDF.impact < 1.0)

    return newDF


def get_ntransits(row, sectorlength=13.7, nsectors=114):
    totalMissionDuration = sectorlength * nsectors
    transitTimes = np.arange(
        row.loc['T0'], totalMissionDuration, row.loc['planetPeriod'])
    bins = sectorlength * np.arange(0, 1 + nsectors)
    inds = np.digitize(transitTimes, bins=bins, right=True)
    inds = inds[inds < nsectors + 1]  # assuming we go 4 years and 5 months
    return np.count_nonzero(inds * row.loc[[str(x) for x in inds]])


def get_ntransits_primary(row, sectorlength=13.7, nsectors=114):
    totalMissionDuration = sectorlength * nsectors
    transitTimes = np.arange(
        row.loc['T0'], totalMissionDuration, row.loc['planetPeriod'])
    if sectorlength < 20:
        bins = sectorlength * np.arange(0, 1 + (26 * 2))
        inds = np.digitize(transitTimes, bins=bins, right=True)
        inds = inds[inds < 1 + (26 * 2)]
    else:
        bins = sectorlength * np.arange(0, 1 + (13 * 2))
        inds = np.digitize(transitTimes, bins=bins, right=True)
        inds = inds[inds < 1 + (13 * 2)]
    return np.count_nonzero(inds * row.loc[[str(x) for x in inds]])


def get_insol(teff, ars):
    p1 = (teff / 5771)**4
    p2 = (215.1 / ars)**2
    return p1 * p2


def make_output_arr(dfx, nsectors):
    # which stars are observed
    obscols = [str(x) for x in range(1, nsectors + 1)]
    dfx.loc[:, 'isObserved'] = dfx.loc[:, obscols].sum(axis=1) > 0

    # how many observed transits
    # this line takes several minutes
    dfx.loc[:, 'Ntransits'] = dfx.progress_apply(get_ntransits, axis=1)

    # how many observed transits in the primary mission
    dfx.loc[:, 'Ntransits_primary'] = dfx.progress_apply(
        get_ntransits_primary, axis=1)

    # get SNR
    dfx.loc[:, 'SNR'] = (dfx.transit_depth_diluted * dfx.duration_correction *
                         np.sqrt(dfx.Ntransits) / dfx.noise_level)
    dfx.loc[:, 'SNR_primary'] = (dfx.transit_depth_diluted * dfx.duration_correction *
                                 np.sqrt(dfx.Ntransits_primary) / dfx.noise_level)

    dfx['needed_for_detection'] = (dfx.transit_depth_diluted * dfx.duration_correction *
                                   np.sqrt(dfx.Ntransits)) / consts['sigma_threshold']
    dfx['detected'] = ((dfx.noise_level < dfx.needed_for_detection) &
                       (dfx.Ntransits >= consts['detect_transits']) & (
                           dfx.planetRadius > 0.0)
                       & dfx.has_transits)

    dfx['needed_for_detection_primary'] = (dfx.transit_depth_diluted * dfx.duration_correction *
                                           np.sqrt(dfx.Ntransits_primary)) / consts['sigma_threshold']
    dfx['detected_primary'] = ((dfx.noise_level < dfx.needed_for_detection_primary) &
                               (dfx.Ntransits_primary >= consts['detect_transits']) & (
                                   dfx.planetRadius > 0.0)
                               & dfx.has_transits)
    dfx.loc[:, 'insol'] = get_insol(dfx.TEFF, dfx.ars)
    dfx.loc[:, 'inOptimisticHZ'] = False
    dfx.loc[(dfx.insol >= 0.32) & (dfx.insol <= 1.78), 'inOptimisticHZ'] = True
    return dfx


if __name__ == '__main__':

    fn = '../data/exo_CTL_08.01xTIC_v8.csv'

    header = [
        'TICID', 'RA', 'DEC', 'PLX', 'ECLONG', 'ECLAT',
        'V', 'J', 'Ks', 'TESSMAG', 'TEFF',
        'RADIUS', 'MASS', 'CONTRATIO', 'PRIORITY'
    ]

    usecols = [0, 13, 14, 21, 26, 27, 30, 42, 46, 60, 64, 70, 72, 84, 87, ]

    print()
    print('doing entire CTL')

    df = pd.read_csv(fn, names=header, usecols=usecols)

    newDF = calculate_planet_properties(df)

    selected = newDF[newDF.has_transits == True]
    selected.to_csv('../data/allCTL8-EM-{}-{}T.csv.bz2'.format(consts['version'], consts['detect_transits']),
                    compression='bz2')

    out_SNE = get_camera_bouma(selected.reset_index(
        drop=True), fieldfile='../data/camera_boresights_SNE-shifted.csv')
    # out_SNSNS = get_camera_bouma(selected.reset_index(
    #     drop=True), fieldfile='../data/camera_boresights_SNSNS.csv')
    # out_SNNSN = get_camera_bouma(selected.reset_index(
    #     drop=True), fieldfile='../data/camera_boresights_SNNSN.csv')
    # out_EC3PO = get_camera_bouma(selected.reset_index(
    #     drop=True), fieldfile='../data/camera_boresights_EC3PO.csv')

    df_out_SNE = pd.DataFrame(out_SNE, columns=[str(
        x) for x in range(1, 1 + out_SNE.shape[1])])
    # df_out_SNSNS = pd.DataFrame(
    #     out_SNSNS, columns=[str(x) for x in range(1, 1 + out_SNE.shape[1])])
    # df_out_SNNSN = pd.DataFrame(
    #     out_SNNSN, columns=[str(x) for x in range(1, 1 + out_SNE.shape[1])])
    # df_out_EC3PO = pd.DataFrame(
    #     out_EC3PO, columns=[str(x) for x in range(1, 1 + out_SNE.shape[1])])

    dfw_SNE = pd.concat([selected.reset_index(drop=True), df_out_SNE], axis=1)
    nsectors = out_SNE.shape[1]
    dfw_SNE = make_output_arr(dfw_SNE, nsectors)
    print('Planets detected in primary + extended mission SNE: {}'.format(
        dfw_SNE[dfw_SNE.detected].shape[0]))
    print('Planets detected in primary mission SNE: {}'.format(
        dfw_SNE[dfw_SNE.detected_primary].shape[0]))
    dfw_SNE.to_csv('../data/obs_SNE-{}-{}T-CTL8.csv.bz2'.format(consts['version'], consts['detect_transits']),
                   compression='bz2')

    # dfw_SNSNS = pd.concat(
    #     [selected.reset_index(drop=True), df_out_SNSNS], axis=1)
    # nsectors = out_SNSNS.shape[1]
    # dfw_SNSNS = make_output_arr(dfw_SNSNS, nsectors)
    # print('Planets detected in primary + extended mission SNSNS: {}'.format(
    #     dfw_SNSNS[dfw_SNSNS.detected].shape[0]))
    # print('Planets detected in primary mission SNSNS: {}'.format(
    #     dfw_SNSNS[dfw_SNSNS.detected_primary].shape[0]))
    # dfw_SNSNS.to_csv('../data/obs_SNSNS-{}-{}T.csv.bz2'.format(consts['version'], consts['detect_transits']),
    #                  compression='bz2')

    # dfw_SNNSN = pd.concat(
    #     [selected.reset_index(drop=True), df_out_SNNSN], axis=1)
    # nsectors = out_SNNSN.shape[1]
    # dfw_SNNSN = make_output_arr(dfw_SNNSN, nsectors)
    # print('Planets detected in primary + extended mission SNNSN: {}'.format(
    #     dfw_SNNSN[dfw_SNNSN.detected].shape[0]))
    # print('Planets detected in primary mission SNNSN: {}'.format(
    #     dfw_SNNSN[dfw_SNE.detected_primary].shape[0]))
    # dfw_SNNSN.to_csv('../data/obs_SNNSN-{}-{}T.csv.bz2'.format(consts['version'], consts['detect_transits']),
    #                  compression='bz2')

    # dfw_EC3PO = pd.concat(
    #     [selected.reset_index(drop=True), df_out_EC3PO], axis=1)
    # nsectors = out_EC3PO.shape[1]
    # dfw_EC3PO = make_output_arr(dfw_EC3PO, nsectors)
    # print('Planets detected in primary + extended mission EC3PO: {}'.format(
    #     dfw_EC3PO[dfw_EC3PO.detected].shape[0]))
    # print('Planets detected in primary mission EC3PO: {}'.format(
    #     dfw_EC3PO[dfw_EC3PO.detected_primary].shape[0]))
    # dfw_EC3PO.to_csv('../data/obs_EC3PO-{}-{}T.csv.bz2'.format(consts['version'], consts['detect_transits']),
    #                  compression='bz2')

    print('doing the final save of all the planets')
    newDF.to_csv('../data/allCTL7-EM-{}-everything-CTL8.csv.bz2'.format(consts['version']),
                 compression='bz2')
