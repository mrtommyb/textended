import numpy as np
import pandas as pd
import sys

from tqdm import tqdm, trange

from astropy.coordinates import SkyCoord
from astropy import units as u
from numpy.random import poisson, beta, uniform
from numpy import array as nparr
import simfuncs

tqdm.pandas()

consts = {
    "sigma_threshold": 7.1,
    "detect_transits": 3,
    "sector_length": 91.3125,
    "version": "v2",
    # "fgk_rate": 2.5,#0.69,#2.5,
    "m_rate": 2.96,
    "ocrMeasurement": "burke",
}

if consts['ocrMeasurement'] == 'bryson':
    consts['fgk_rate'] = 0.69
elif consts['ocrMeasurement'] == 'burke':
    consts['fgk_rate'] = 2.5
elif consts['ocrMeasurement'] == 'LUVOIR':
    consts['fgk_rate'] = 0.05


def get_quarters(strategy="k1k2"):
    # arrary runs from Q1 2009 - Q4 2030
    quartersObserved = np.zeros(4 * 31)

    if (strategy == "k1") or (strategy == "k1k2"):
        # assume that Kepler lasted 16 quarters
        quartersObserved[1 : 1 + 1 + 16] = 1

    if (strategy == "k2") or (strategy == "k1k2"):
        # assume that Kepler lasted 16 quarters
        quartersObserved[-20:] = 1

    return quartersObserved


def calculate_planet_properties(df):
    df["isMdwarf"] = np.where((df.teff < 3900) & (df.radius < 0.6), True, False)
    df["isGiant"] = np.zeros(df.shape[0], dtype="bool")  # assume all dwarfs

    df["cosi"] = pd.Series(np.random.random(size=df.shape[0]), name="cosi")
    df["noise_level"] = simfuncs.kepler_noise_1h_quiet(df.kepmag)

    np_fgk = poisson(lam=consts["fgk_rate"], size=df.shape[0])
    if consts['ocrMeasurement'] == 'LUVOIR':
        np_fgk = np.where((df.teff > 5300) & (df.teff < 6000), np_fgk, 0)
    np_m = poisson(lam=consts["m_rate"], size=df.shape[0])
    df["Nplanets"] = pd.Series(
        np.where(df.isMdwarf, np_m, np_fgk), name="Nplanets"
    )

    starID = 0  # ???
    newDF, starID = simfuncs.make_allplanets_df_vec_extrap_kepler(
        df, starID, ocrMeasurement=consts["ocrMeasurement"]
    )
    newDF = newDF.assign(
        T0=pd.Series(
            uniform(0, 1, size=newDF.shape[0]) * newDF.loc[:, "planetPeriod"]
        )
    )

    newDF["ars"] = simfuncs.per2ars(
        newDF.planetPeriod, newDF.mass, newDF.radius
    )
    # ecc dist from Van Eylen 2015
    newDF["ecc"] = pd.Series(beta(1.03, 13.6, size=newDF.shape[0]), name="ecc")
    newDF["omega"] = pd.Series(
        uniform(-np.pi, np.pi, size=newDF.shape[0]), name="omega"
    )
    newDF["rprs"] = simfuncs.get_rprs(newDF.planetRadius, newDF.radius)
    newDF["impact"] = (
        newDF.cosi
        * newDF.ars
        * ((1 - newDF.ecc ** 2) / 1 + newDF.ecc * np.sin(newDF.omega))
    )  # cite Winn
    newDF["duration"] = simfuncs.get_duration(
        newDF.planetPeriod,
        newDF.ars,
        cosi=newDF.cosi,
        b=newDF.impact,
        rprs=newDF.rprs,
    )  # cite Winn

    # correction for CDPP because transit dur != 1 hour
    newDF["duration_correction"] = np.sqrt(newDF.duration * 24.0)
    newDF["transit_depth"] = simfuncs.get_transit_depth(
        newDF.planetRadius, newDF.radius
    )

    newDF["transit_depth_diluted"] = newDF["transit_depth"] * (
        newDF.Crowdingseason0
    )

    newDF["has_transits"] = (newDF.ars > 1.0) & (newDF.impact < 1.0)

    return newDF


def get_ntransits(row, sectorlength=91.3125, nsectors=124):
    totalMissionDuration = sectorlength * nsectors
    transitTimes = np.arange(
        row.loc["T0"], totalMissionDuration, row.loc["planetPeriod"]
    )
    bins = sectorlength * np.arange(0, 1 + nsectors)
    inds = np.digitize(transitTimes, bins=bins, right=True)
    inds = inds[inds < nsectors + 1]  # assuming we go 4 years and 5 months
    return np.count_nonzero(inds * row.loc[[str(x) for x in inds]])


def get_ntransits_primary(row, sectorlength=91.3125, nsectors=124):
    totalMissionDuration = sectorlength * nsectors
    transitTimes = np.arange(
        row.loc["T0"], totalMissionDuration, row.loc["planetPeriod"]
    )

    bins = sectorlength * np.arange(1, 1 + 16 + 1)
    inds = np.digitize(transitTimes, bins=bins, right=True)
    inds = inds[inds < 1 + 16 ]
    return np.count_nonzero(inds * row.loc[[str(x) for x in inds]])


def get_insol(teff, ars):
    p1 = (teff / 5771) ** 4
    p2 = (215.1 / ars) ** 2
    return p1 * p2


def make_output_arr(dfx, nsectors):
    # which stars are observed
    obscols = [str(x) for x in range(1, nsectors + 1)]
    dfx.loc[:, "isObserved"] = dfx.loc[:, obscols].sum(axis=1) > 0

    # how many observed transits
    # this line takes several minutes
    dfx.loc[:, "Ntransits"] = dfx.progress_apply(get_ntransits, axis=1)

    # how many observed transits in the primary mission
    dfx.loc[:, "Ntransits_primary"] = dfx.progress_apply(
        get_ntransits_primary, axis=1
    )

    # get SNR
    dfx.loc[:, "SNR"] = (
        dfx.transit_depth_diluted
        * dfx.duration_correction
        * np.sqrt(dfx.Ntransits)
        / dfx.noise_level
    )
    dfx.loc[:, "SNR_primary"] = (
        dfx.transit_depth_diluted
        * dfx.duration_correction
        * np.sqrt(dfx.Ntransits_primary)
        / dfx.noise_level
    )

    dfx["needed_for_detection"] = (
        dfx.transit_depth_diluted
        * dfx.duration_correction
        * np.sqrt(dfx.Ntransits)
    ) / consts["sigma_threshold"]
    dfx["detected"] = (
        (dfx.noise_level < dfx.needed_for_detection)
        & (dfx.Ntransits >= consts["detect_transits"])
        & (dfx.planetRadius > 0.0)
        & dfx.has_transits
    )

    dfx["needed_for_detection_primary"] = (
        dfx.transit_depth_diluted
        * dfx.duration_correction
        * np.sqrt(dfx.Ntransits_primary)
    ) / consts["sigma_threshold"]
    dfx["detected_primary"] = (
        (dfx.noise_level < dfx.needed_for_detection_primary)
        & (dfx.Ntransits_primary >= consts["detect_transits"])
        & (dfx.planetRadius > 0.0)
        & dfx.has_transits
    )
    dfx.loc[:, "insol"] = get_insol(dfx.teff, dfx.ars)
    dfx.loc[:, "inOptimisticHZ"] = False
    dfx.loc[(dfx.insol >= 0.32) & (dfx.insol <= 1.78), "inOptimisticHZ"] = True
    dfx.loc[:, "inZetaEarth"] = False
    dfx.loc[
        (dfx.planetRadius >= 0.8)
        & (dfx.planetRadius <= 1.2)
        & (dfx.planetPeriod <= 438.3)
        & (dfx.planetPeriod >= 292.2),
        "inZetaEarth",
    ] = True
    # dfx.loc[:, "inSAG13"] = False
    # dfx.loc[
    #     (dfx.planetRadius >= 0.5)
    #     & (dfx.planetRadius <= 1.5)
    #     & (dfx.planetPeriod <= 860)
    #     & (dfx.planetPeriod >= 237),
    #     "inSAG13",
    # ] = True
    return dfx


if __name__ == "__main__":
    fn = "../data/bryson/dr25_stellar_berger2019_clean_GK_withContratio.txt"

    header = [
        "kepid",
        "teff",
        "mass",
        "radius",
        "kepmag",
        "dist",
        "nkoi",
        "ra",
        "dec",
        "st_quarters",
        "jmag",
        "hmag",
        "kmag",
        "rrmscdpp06p0",
        "Crowdingseason0",
    ]

    usecols = [0, 2, 11, 14, 21, 26, 30, 35, 36, 37, 41, 43, 45, 70, 158]

    print()
    print("doing entire CTL")

    df = pd.read_csv(fn, names=header, usecols=usecols, skiprows=1)
    newDF = calculate_planet_properties(df)

    selected = newDF[newDF.has_transits == True]
    selected.to_csv(
        "../data/KeplerBryson-EM-{}-{}T.csv.bz2".format(
            consts["version"], consts["detect_transits"]
        ),
        compression="bz2",
    )
    selected.to_csv(
        "../data/KeplerBryson-EM-{}-{}T-{}.csv".format(
            consts["version"], consts["detect_transits"], consts["ocrMeasurement"],
        )
    )
    out_kepler = np.tile(get_quarters(), [selected.shape[0], 1])
    df_out_kepler = pd.DataFrame(
        out_kepler, columns=[str(x) for x in range(1, 1 + out_kepler.shape[1])]
    )
    dfw_kepler = pd.concat(
        [selected.reset_index(drop=True), df_out_kepler], axis=1
    )
    nsectors = out_kepler.shape[1]
    dfw_kepler = make_output_arr(dfw_kepler, nsectors)

    print(
        "Planets detected in primary + extended mission: {}".format(
            dfw_kepler[dfw_kepler.detected].shape[0]
        )
    )
    print(
        "Planets detected in primary mission SNE: {}".format(
            dfw_kepler[dfw_kepler.detected_primary].shape[0]
        )
    )
    dfw_kepler.to_csv(
        "../data/obs_kepler-{}-{}T-{}.csv".format(
            consts["version"], consts["detect_transits"], consts["ocrMeasurement"],
        )
    )

    q1 = np.array([])
    q2 = np.array([])
    q3 = np.array([])
    q4 = np.array([])
    q5 = np.array([])
    q6 = np.array([])
    for i in trange(500):
        newDF = calculate_planet_properties(df)

        selected = newDF[newDF.has_transits == True]
        out_kepler = np.tile(get_quarters(), [selected.shape[0], 1])
        df_out_kepler = pd.DataFrame(
            out_kepler,
            columns=[str(x) for x in range(1, 1 + out_kepler.shape[1])],
        )
        dfw_kepler = pd.concat(
            [selected.reset_index(drop=True), df_out_kepler], axis=1
        )
        nsectors = out_kepler.shape[1]
        dfw_kepler = make_output_arr(dfw_kepler, nsectors)

        q1 = np.r_[q1, dfw_kepler[dfw_kepler.detected_primary].shape[0]]
        q2 = np.r_[q2, dfw_kepler[dfw_kepler.detected].shape[0]]

        q3 = np.r_[
            q3,
            dfw_kepler[
                dfw_kepler.detected_primary & (dfw_kepler.inZetaEarth)
            ].shape[0],
        ]
        q4 = np.r_[
            q4,
            dfw_kepler[(dfw_kepler.detected) & (dfw_kepler.inZetaEarth)].shape[
                0
            ],
        ]

        # q5 = np.r_[
        #     q5,
        #     dfw_kepler[
        #         dfw_kepler.detected_primary & (dfw_kepler.inSAG13)
        #     ].shape[0],
        # ]
        # q6 = np.r_[
        #     q6,
        #     dfw_kepler[(dfw_kepler.detected) & (dfw_kepler.inSAG13)].shape[0],
        # ]

        dfw_kepler.to_csv(
        "../data/bryson/{}/obs_kepler-{}-{}T-{}-n{}.csv".format(consts["ocrMeasurement"],
        consts["version"], consts["detect_transits"], consts["ocrMeasurement"],
        i)
        )
