import numpy as np
from numpy import random
from scipy.interpolate import interp1d
import pandas as pd

msun = 1.9891e30
rsun = 695500000.0
G = 6.67384e-11
AU = 149597870700.0


def component_noise(tessmag, readmod=1, zodimod=1):
    sys = 59.785
    star_mag_level, star_noise_level = np.array(
        [
            [4.3885191347753745, 12.090570910640581],
            [12.023294509151416, 467.96434635620614],
            [17.753743760399338, 7779.603209291808],
        ]
    ).T
    star_pars = np.polyfit(star_mag_level, np.log10(star_noise_level), 1)
    zodi_mag_level, zodi_noise_level = np.array(
        [
            [8.686356073211314, 18.112513551189224],
            [13.08901830282862, 688.2812796087189],
            [16.68801996672213, 19493.670323892282],
        ]
    ).T
    zodi_pars = np.polyfit(zodi_mag_level, np.log10(zodi_noise_level), 1)
    read_mag_level, read_noise_level = np.array(
        [
            [8.476705490848586, 12.31474807751376],
            [13.019134775374376, 522.4985702369348],
            [17.841098169717142, 46226.777232915076],
        ]
    ).T
    read_pars = np.polyfit(read_mag_level, np.log10(read_noise_level), 1)

    c1, c2, c3, c4 = (
        10 ** (tessmag * star_pars[0] + star_pars[1]),
        10 ** (tessmag * zodi_pars[0] + zodi_pars[1]),
        10 ** (tessmag * read_pars[0] + read_pars[1]),
        sys,
    )

    return np.sqrt(
        c1 ** 2 + (readmod * c2) ** 2 + (zodimod * c3) ** 2 + c4 ** 2
    )


def rndm(a, b, g, size=1):
    """Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b"""
    r = np.random.random(size=size)
    ag, bg = a ** g, b ** g
    return (ag + (bg - ag) * r) ** (1.0 / g)


def Fressin13_select_extrap(nselect=1):
    # create a pot for dressing numbers (balls)
    balls = np.array([])
    # pot 1 contains rp=0.8-0.1.25, p=0.8-2
    p1 = np.zeros(180) + 1
    # pot 2 contains rp=1.25-2.0, p=0.8-2
    p2 = np.zeros(170) + 2
    # pot 3 contains rp=2-4, p=0.8-2
    p3 = np.zeros(35) + 3
    # pot 4 contains rp=4-6, p=0.8-2
    p4 = np.zeros(4) + 4
    # pot 5 contains rp=6-22, p=0.8-2
    p5 = np.zeros(15) + 5

    # pot 6 contains rp=0.8-0.1.25, p=2-3.4
    p6 = np.zeros(610) + 6
    # pot 7 contains rp=1.25-2.0, p=2-3.4
    p7 = np.zeros(740) + 7
    # pot 8 contains rp=2-4, p=2-3.4
    p8 = np.zeros(180) + 8
    # pot 9 contains rp=4-6, p=2-3.4
    p9 = np.zeros(6) + 9
    # pot 10 contains rp=6-22, p=2-3.4
    p10 = np.zeros(67) + 10

    # pot 11 contains rp=0.8-0.1.25, p=3.4-5.9
    p11 = np.zeros(1720) + 11
    # pot 12 contains rp=1.25-2.0, p=3.4-5.9
    p12 = np.zeros(1490) + 12
    # pot 13 contains rp=2-4, p=3.4-5.9
    p13 = np.zeros(730) + 13
    # pot 14 contains rp=4-6, p=3.4-5.9
    p14 = np.zeros(110) + 14
    # pot 15 contains rp=6-22, p=3.4-5.9
    p15 = np.zeros(170) + 15

    # pot 16 contains rp=0.8-0.1.25, p=5.9-10
    p16 = np.zeros(2700) + 16
    # pot 17 contains rp=1.25-2.0, p=5.9-10
    p17 = np.zeros(2900) + 17
    # pot 18 contains rp=2-4, p=5.9-10
    p18 = np.zeros(1930) + 18
    # pot 19 contains rp=4-6, p=5.9-10
    p19 = np.zeros(91) + 19
    # pot 20 contains rp=6-22, p=5.9-10
    p20 = np.zeros(180) + 20

    # pot 21 contains rp=0.8-0.1.25, p=10-17
    p21 = np.zeros(2700) + 21
    # pot 22 contains rp=1.25-2.0, p=10-17
    p22 = np.zeros(4300) + 22
    # pot 23 contains rp=2-4, p=10-17
    p23 = np.zeros(3670) + 23
    # pot 24 contains rp=4-6, p=10-17
    p24 = np.zeros(290) + 24
    # pot 25 contains rp=6-22, p=10-17
    p25 = np.zeros(270) + 25

    # pot 26 contains rp=0.8-0.1.25, p=17-29
    p26 = np.zeros(2930) + 26
    # pot 27 contains rp=1.25-2.0, p=17-29
    p27 = np.zeros(4490) + 27
    # pot 28 contains rp=2-4, p=17-29
    p28 = np.zeros(5290) + 28
    # pot 29 contains rp=4-6, p=17-29
    p29 = np.zeros(320) + 29
    # pot 30 contains rp=6-22, p=17-29
    p30 = np.zeros(230) + 30

    # pot 31 contains rp=0.8-0.1.25, p=29-50
    p31 = np.zeros(4080) + 31
    # pot 32 contains rp=1.25-2.0, p=29-50
    p32 = np.zeros(5290) + 32
    # pot 33 contains rp=2-4, p=29-50
    p33 = np.zeros(6450) + 33
    # pot 34 contains rp=4-6, p=29-50
    p34 = np.zeros(490) + 34
    # pot 35 contains rp=6-22, p=29-50
    p35 = np.zeros(350) + 35

    # pot 36 contains rp=0.8-0.1.25, p=50-85
    p36 = np.zeros(3460) + 36
    # pot 37 contains rp=1.25-2.0, p=50-85
    p37 = np.zeros(3660) + 37
    # pot 38 contains rp=2-4, p=50-85
    p38 = np.zeros(5250) + 38
    # pot 39 contains rp=4-6, p=50-85
    p39 = np.zeros(660) + 39
    # pot 40 contains rp=6-22, p=50-85
    p40 = np.zeros(710) + 40

    # pot 36 contains rp=0.8-0.1.25, p=50-85
    p41 = np.zeros(3460) + 41
    # pot 37 contains rp=1.25-2.0, p=50-85
    p42 = np.zeros(3660) + 42
    # pot 38 contains rp=2-4, p=50-85
    p43 = np.zeros(5250) + 43
    # pot 39 contains rp=4-6, p=50-85
    p44 = np.zeros(660) + 44
    # pot 40 contains rp=6-22, p=50-85
    p45 = np.zeros(710) + 45

    # pot 36 contains rp=0.8-0.1.25, p=50-85
    p46 = np.zeros(3460) + 46
    # pot 37 contains rp=1.25-2.0, p=50-85
    p47 = np.zeros(3660) + 47
    # pot 38 contains rp=2-4, p=50-85
    p48 = np.zeros(5250) + 48
    # pot 39 contains rp=4-6, p=50-85
    p49 = np.zeros(660) + 49
    # pot 40 contains rp=6-22, p=50-85
    p50 = np.zeros(710) + 50

    # pot 36 contains rp=0.8-0.1.25, p=50-85
    p51 = np.zeros(3460) + 51
    # pot 37 contains rp=1.25-2.0, p=50-85
    p52 = np.zeros(3660) + 52
    # pot 38 contains rp=2-4, p=50-85
    p53 = np.zeros(5250) + 53
    # pot 39 contains rp=4-6, p=50-85
    p54 = np.zeros(660) + 54
    # pot 40 contains rp=6-22, p=50-85
    p55 = np.zeros(710) + 55

    balls = np.r_[
        balls,
        p1,
        p2,
        p3,
        p4,
        p5,
        p6,
        p7,
        p8,
        p9,
        p10,
        p11,
        p12,
        p13,
        p14,
        p15,
        p16,
        p17,
        p18,
        p19,
        p20,
        p21,
        p22,
        p23,
        p24,
        p25,
        p26,
        p27,
        p28,
        p29,
        p30,
        p31,
        p32,
        p33,
        p34,
        p35,
        p36,
        p37,
        p38,
        p39,
        p40,
        p41,
        p42,
        p43,
        p44,
        p45,
        p46,
        p47,
        p48,
        p49,
        p50,
        p51,
        p52,
        p53,
        p54,
        p55,
    ]

    # lookup for what the balls mean
    # outputs radlow, radhigh, Plow, Phigh
    ball_lookup = {
        0: [0.0, 0.0, 0.0, 0.0],
        1: [0.8, 1.25, 0.8, 2.0],
        2: [1.25, 2.0, 0.8, 2.0],
        3: [2.0, 4.0, 0.8, 2.0],
        4: [4.0, 6.0, 0.8, 2.0],
        5: [6.0, 22.0, 0.8, 2.0],
        6: [0.8, 1.25, 2.0, 3.4],
        7: [1.25, 2.0, 2.0, 3.4],
        8: [2.0, 4.0, 2.0, 3.4],
        9: [4.0, 6.0, 2.0, 3.4],
        10: [6.0, 22.0, 2.0, 3.4],
        11: [0.8, 1.25, 3.4, 5.9],
        12: [1.25, 2.0, 3.4, 5.9],
        13: [2.0, 4.0, 3.4, 5.9],
        14: [4.0, 6.0, 3.4, 5.9],
        15: [6.0, 22.0, 3.4, 5.9],
        16: [0.8, 1.25, 5.9, 10.0],
        17: [1.25, 2.0, 5.9, 10.0],
        18: [2.0, 4.0, 5.9, 10.0],
        19: [4.0, 6.0, 5.9, 10.0],
        20: [6.0, 22.0, 5.9, 10.0],
        21: [0.8, 1.25, 10.0, 17.0],
        22: [1.25, 2.0, 10.0, 17.0],
        23: [2.0, 4.0, 10.0, 17.0],
        24: [4.0, 6.0, 10.0, 17.0],
        25: [6.0, 22.0, 10.0, 17.0],
        26: [0.8, 1.25, 17.0, 29.0],
        27: [1.25, 2.0, 17.0, 29.0],
        28: [2.0, 4.0, 17.0, 29.0],
        29: [4.0, 6.0, 17.0, 29.0],
        30: [6.0, 22.0, 17.0, 29.0],
        31: [0.8, 1.25, 29.0, 50.0],
        32: [1.25, 2.0, 29.0, 50.0],
        33: [2.0, 4.0, 29.0, 50.0],
        34: [4.0, 6.0, 29.0, 50.0],
        35: [6.0, 22.0, 29.0, 50.0],
        36: [0.8, 1.25, 50.0, 85.0],
        37: [1.25, 2.0, 50.0, 85.0],
        38: [2.0, 4.0, 50.0, 85.0],
        39: [4.0, 6.0, 50.0, 85.0],
        40: [6.0, 22.0, 50.0, 85.0],
        41: [0.8, 1.25, 50.0, 150.0],
        42: [1.25, 2.0, 50.0, 150.0],
        43: [2.0, 4.0, 50.0, 150.0],
        44: [4.0, 6.0, 50.0, 150.0],
        45: [6.0, 22.0, 50.0, 150.0],
        46: [0.8, 1.25, 150.0, 270.0],
        47: [1.25, 2.0, 150.0, 270.0],
        48: [2.0, 4.0, 150.0, 270.0],
        49: [4.0, 6.0, 150.0, 270.0],
        50: [6.0, 22.0, 150.0, 270.0],
        51: [0.8, 1.25, 270.0, 480.0],
        52: [1.25, 2.0, 270.0, 480.0],
        53: [2.0, 4.0, 270.0, 480.0],
        54: [4.0, 6.0, 270.0, 480.0],
        55: [6.0, 22.0, 270.0, 480.0],
    }

    rsamps = random.choice(balls, size=nselect)
    radius = np.zeros(nselect)
    period = np.zeros(nselect)
    for i, samp in enumerate(rsamps):
        rl, rh, pl, ph = ball_lookup[samp]

        if samp in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
            # check for giant planets
            # if a giant planet than draw power law
            radius[i] = rndm(6, 22, -1.7)
        else:
            radius[i] = random.uniform(low=rl, high=rh)

        period[i] = random.uniform(low=pl, high=ph)

    return radius, period


def Dressing15_select_extrap(nselect=1):
    """
    period bins = 0.5, 0.91, 1.66, 3.02, 5.49, 10.0, 18.2, 33.1, 60.3, 110., 200.
    """
    # create a pot for dressing numbers (balls)
    balls = np.array([])
    # pot 1 contains rp=0.5-1.0, p=0.5-0.91
    p1 = np.zeros(400) + 1
    # pot 2 contains rp=1.0-1.5, p=0.5-0.91
    p2 = np.zeros(460) + 2
    # pot 3 contains rp=1.5-2.0, p=0.5-0.91
    p3 = np.zeros(61) + 3
    # pot 4 contains rp=2.0-2.5, p=0.5-0.91
    p4 = np.zeros(2) + 4
    # pot 5 contains rp=2.5-3.0, p=0.5-0.91
    p5 = np.zeros(0) + 5
    # pot 6 contains rp=3.0-3.5, p=0.5-0.91
    p6 = np.zeros(0) + 6
    # pot 7 contains rp=3.5-4.0, p=0.5-0.91
    p7 = np.zeros(0) + 7
    # pot 1 contains rp=0.5-1.0, p=0.91, 1.66
    p8 = np.zeros(1500) + 8
    # pot 2 contains rp=1.0-1.5, p=0.91, 1.66
    p9 = np.zeros(1400) + 9
    # pot 3 contains rp=1.5-2.0, p=0.91, 1.66
    p10 = np.zeros(270) + 10
    # pot 4 contains rp=2.0-2.5, p=0.91, 1.66
    p11 = np.zeros(9) + 11
    # pot 5 contains rp=2.5-3.0, p=0.91, 1.66
    p12 = np.zeros(4) + 12
    # pot 6 contains rp=3.0-3.5, p=0.91, 1.66
    p13 = np.zeros(6) + 13
    # pot 7 contains rp=3.5-4.0, p=0.91, 1.66
    p14 = np.zeros(8) + 14
    # pot 1 contains rp=0.5-1.0, p=1.66, 3.02
    p15 = np.zeros(4400) + 15
    # pot 2 contains rp=1.0-1.5, p=1.66, 3.02
    p16 = np.zeros(3500) + 16
    # pot 3 contains rp=1.5-2.0, p=1.66, 3.02
    p17 = np.zeros(1200) + 17
    # pot 4 contains rp=2.0-2.5, p=1.66, 3.02
    p18 = np.zeros(420) + 18
    # pot 5 contains rp=2.5-3.0, p=1.66, 3.02
    p19 = np.zeros(230) + 19
    # pot 6 contains rp=3.0-3.5, p=1.66, 3.02
    p20 = np.zeros(170) + 20
    # pot 7 contains rp=3.5-4.0, p=1.66, 3.02
    p21 = np.zeros(180) + 21
    # pot 1 contains rp=0.5-1.0, p=3.02, 5.49
    p22 = np.zeros(5500) + 22
    # pot 2 contains rp=1.0-1.5, p=3.02, 5.49
    p23 = np.zeros(5700) + 23
    # pot 3 contains rp=1.5-2.0, p=3.02, 5.49
    p24 = np.zeros(2500) + 24
    # pot 4 contains rp=2.0-2.5, p=3.02, 5.49
    p25 = np.zeros(1800) + 25
    # pot 5 contains rp=2.5-3.0, p=3.02, 5.49
    p26 = np.zeros(960) + 26
    # pot 6 contains rp=3.0-3.5, p=3.02, 5.49
    p27 = np.zeros(420) + 27
    # pot 7 contains rp=3.5-4.0, p=3.02, 5.49
    p28 = np.zeros(180) + 28
    # pot 1 contains rp=0.5-1.0, p=5.49, 10.0
    p29 = np.zeros(10000) + 29
    # pot 2 contains rp=1.0-1.5, p=5.49, 10.0
    p30 = np.zeros(10000) + 30
    # pot 3 contains rp=1.5-2.0, p=5.49, 10.0
    p31 = np.zeros(6700) + 31
    # pot 4 contains rp=2.0-2.5, p=5.49, 10.0
    p32 = np.zeros(6400) + 32
    # pot 5 contains rp=2.5-3.0, p=5.49, 10.0
    p33 = np.zeros(2700) + 33
    # pot 6 contains rp=3.0-3.5, p=5.49, 10.0
    p34 = np.zeros(1100) + 34
    # pot 7 contains rp=3.5-4.0, p=5.49, 10.0
    p35 = np.zeros(360) + 35
    # pot 1 contains rp=0.5-1.0, p=10.0, 18.2
    p36 = np.zeros(12000) + 36
    # pot 2 contains rp=1.0-1.5, p=10.0, 18.2
    p37 = np.zeros(13000) + 37
    # pot 3 contains rp=1.5-2.0, p=10.0, 18.2
    p38 = np.zeros(13000) + 38
    # pot 4 contains rp=2.0-2.5, p=10.0, 18.2
    p39 = np.zeros(9300) + 39
    # pot 5 contains rp=2.5-3.0, p=10.0, 18.2
    p40 = np.zeros(3800) + 40
    # pot 6 contains rp=3.0-3.5, p=10.0, 18.2
    p41 = np.zeros(1400) + 41
    # pot 7 contains rp=3.5-4.0, p=10.0, 18.2
    p42 = np.zeros(510) + 42
    # pot 1 contains rp=0.5-1.0, p=18.2, 33.1
    p43 = np.zeros(11000) + 43
    # pot 2 contains rp=1.0-1.5, p=18.2, 33.1
    p44 = np.zeros(16000) + 44
    # pot 3 contains rp=1.5-2.0, p=18.2, 33.1
    p45 = np.zeros(14000) + 45
    # pot 4 contains rp=2.0-2.5, p=18.2, 33.1
    p46 = np.zeros(10000) + 46
    # pot 5 contains rp=2.5-3.0, p=18.2, 33.1
    p47 = np.zeros(4600) + 47
    # pot 6 contains rp=3.0-3.5, p=18.2, 33.1
    p48 = np.zeros(810) + 48
    # pot 7 contains rp=3.5-4.0, p=18.2, 33.1
    p49 = np.zeros(320) + 49
    # pot 1 contains rp=0.5-1.0, p=33.1, 60.3
    p50 = np.zeros(6400) + 50
    # pot 2 contains rp=1.0-1.5, p=33.1, 60.3
    p51 = np.zeros(6400) + 51
    # pot 3 contains rp=1.5-2.0, p=33.1, 60.3
    p52 = np.zeros(12000) + 52
    # pot 4 contains rp=2.0-2.5, p=33.1, 60.3
    p53 = np.zeros(12000) + 53
    # pot 5 contains rp=2.5-3.0, p=33.1, 60.3
    p54 = np.zeros(5800) + 54
    # pot 6 contains rp=3.0-3.5, p=33.1, 60.3
    p55 = np.zeros(1600) + 55
    # pot 7 contains rp=3.5-4.0, p=33.1, 60.3
    p56 = np.zeros(210) + 56
    # pot 1 contains rp=0.5-1.0, p=60.3, 110.
    p57 = np.zeros(10000) + 57
    # pot 2 contains rp=1.0-1.5, p=60.3, 110.
    p58 = np.zeros(10000) + 58
    # pot 3 contains rp=1.5-2.0, p=60.3, 110.
    p59 = np.zeros(8300) + 59
    # pot 4 contains rp=2.0-2.5, p=60.3, 110.
    p60 = np.zeros(9600) + 60
    # pot 5 contains rp=2.5-3.0, p=60.3, 110.
    p61 = np.zeros(4200) + 61
    # pot 6 contains rp=3.0-3.5, p=60.3, 110.
    p62 = np.zeros(1700) + 62
    # pot 7 contains rp=3.5-4.0, p=60.3, 110.
    p63 = np.zeros(420) + 63
    # pot 1 contains rp=0.5-1.0, p=110., 200.
    p64 = np.zeros(19000) + 64
    # pot 2 contains rp=1.0-1.5, p=110., 200.
    p65 = np.zeros(19000) + 65
    # pot 3 contains rp=1.5-2.0, p=110., 200.
    p66 = np.zeros(10000) + 66
    # pot 4 contains rp=2.0-2.5, p=110., 200.
    p67 = np.zeros(4500) + 67
    # pot 5 contains rp=2.5-3.0, p=110., 200.
    p68 = np.zeros(1100) + 68
    # pot 6 contains rp=3.0-3.5, p=110., 200.
    p69 = np.zeros(160) + 69
    # pot 7 contains rp=3.5-4.0, p=110., 200.
    p70 = np.zeros(80) + 70
    # pot 1 contains rp=0.5-1.0, p=110., 200.
    p71 = np.zeros(19000) + 71
    # pot 2 contains rp=1.0-1.5, p=110., 200.
    p72 = np.zeros(19000) + 72
    # pot 3 contains rp=1.5-2.0, p=110., 200.
    p73 = np.zeros(10000) + 73
    # pot 4 contains rp=2.0-2.5, p=110., 200.
    p74 = np.zeros(4500) + 74
    # pot 5 contains rp=2.5-3.0, p=110., 200.
    p75 = np.zeros(1100) + 75
    # pot 6 contains rp=3.0-3.5, p=110., 200.
    p76 = np.zeros(160) + 76
    # pot 7 contains rp=3.5-4.0, p=110., 200.
    p77 = np.zeros(80) + 77

    balls = np.r_[
        balls,
        p1,
        p2,
        p3,
        p4,
        p5,
        p6,
        p7,
        p8,
        p9,
        p10,
        p11,
        p12,
        p13,
        p14,
        p15,
        p16,
        p17,
        p18,
        p19,
        p20,
        p21,
        p22,
        p23,
        p24,
        p25,
        p26,
        p27,
        p28,
        p29,
        p30,
        p31,
        p32,
        p33,
        p34,
        p35,
        p36,
        p37,
        p38,
        p39,
        p40,
        p41,
        p42,
        p43,
        p44,
        p45,
        p46,
        p47,
        p48,
        p49,
        p50,
        p51,
        p52,
        p53,
        p54,
        p55,
        p56,
        p57,
        p58,
        p59,
        p60,
        p61,
        p62,
        p63,
        p64,
        p65,
        p66,
        p67,
        p68,
        p69,
        p70,
        p71,
        p72,
        p73,
        p74,
        p75,
        p76,
        p77,
    ]

    # lookup for what the balls mean
    # outputs radlow, radhigh, Plow, Phigh
    # 0.5, 0.91, 1.66, 3.02, 5.49, 10.0, 18.2, 33.1, 60.3, 110., 200.
    ball_lookup = {
        1: [0.5, 1.0, 0.5, 0.91],
        2: [1.0, 1.5, 0.5, 0.91],
        3: [1.5, 2.0, 0.5, 0.91],
        4: [2.0, 2.5, 0.5, 0.91],
        5: [2.5, 3.0, 0.5, 0.91],
        6: [3.0, 3.5, 0.5, 0.91],
        7: [3.5, 4.0, 0.5, 0.91],
        8: [0.5, 1.0, 0.91, 1.66],
        9: [1.0, 1.5, 0.91, 1.66],
        10: [1.5, 2.0, 0.91, 1.66],
        11: [2.0, 2.5, 0.91, 1.66],
        12: [2.5, 3.0, 0.91, 1.66],
        13: [3.0, 3.5, 0.91, 1.66],
        14: [3.5, 4.0, 0.91, 1.66],
        15: [0.5, 1.0, 1.66, 3.02],
        16: [1.0, 1.5, 1.66, 3.02],
        17: [1.5, 2.0, 1.66, 3.02],
        18: [2.0, 2.5, 1.66, 3.02],
        19: [2.5, 3.0, 1.66, 3.02],
        20: [3.0, 3.5, 1.66, 3.02],
        21: [3.5, 4.0, 1.66, 3.02],
        22: [0.5, 1.0, 3.02, 5.49],
        23: [1.0, 1.5, 3.02, 5.49],
        24: [1.5, 2.0, 3.02, 5.49],
        25: [2.0, 2.5, 3.02, 5.49],
        26: [2.5, 3.0, 3.02, 5.49],
        27: [3.0, 3.5, 3.02, 5.49],
        28: [3.5, 4.0, 3.02, 5.49],
        29: [0.5, 1.0, 5.49, 10.0],
        30: [1.0, 1.5, 5.49, 10.0],
        31: [1.5, 2.0, 5.49, 10.0],
        32: [2.0, 2.5, 5.49, 10.0],
        33: [2.5, 3.0, 5.49, 10.0],
        34: [3.0, 3.5, 5.49, 10.0],
        35: [3.5, 4.0, 5.49, 10.0],
        36: [0.5, 1.0, 10.0, 18.2],
        37: [1.0, 1.5, 10.0, 18.2],
        38: [1.5, 2.0, 10.0, 18.2],
        39: [2.0, 2.5, 10.0, 18.2],
        40: [2.5, 3.0, 10.0, 18.2],
        41: [3.0, 3.5, 10.0, 18.2],
        42: [3.5, 4.0, 10.0, 18.2],
        43: [0.5, 1.0, 18.2, 33.1],
        44: [1.0, 1.5, 18.2, 33.1],
        45: [1.5, 2.0, 18.2, 33.1],
        46: [2.0, 2.5, 18.2, 33.1],
        47: [2.5, 3.0, 18.2, 33.1],
        48: [3.0, 3.5, 18.2, 33.1],
        49: [3.5, 4.0, 18.2, 33.1],
        50: [0.5, 1.0, 33.1, 60.3],
        51: [1.0, 1.5, 33.1, 60.3],
        52: [1.5, 2.0, 33.1, 60.3],
        53: [2.0, 2.5, 33.1, 60.3],
        54: [2.5, 3.0, 33.1, 60.3],
        55: [3.0, 3.5, 33.1, 60.3],
        56: [3.5, 4.0, 33.1, 60.3],
        57: [0.5, 1.0, 60.3, 110.0],
        58: [1.0, 1.5, 60.3, 110.0],
        59: [1.5, 2.0, 60.3, 110.0],
        60: [2.0, 2.5, 60.3, 110.0],
        61: [2.5, 3.0, 60.3, 110.0],
        62: [3.0, 3.5, 60.3, 110.0],
        63: [3.5, 4.0, 60.3, 110.0],
        64: [0.5, 1.0, 110.0, 200.0],
        65: [1.0, 1.5, 110.0, 200.0],
        66: [1.5, 2.0, 110.0, 200.0],
        67: [2.0, 2.5, 110.0, 200.0],
        68: [2.5, 3.0, 110.0, 200.0],
        69: [3.0, 3.5, 110.0, 200.0],
        70: [3.5, 4.0, 110.0, 200.0],
        71: [0.5, 1.0, 200.0, 365.0],
        72: [1.0, 1.5, 200.0, 365.0],
        73: [1.5, 2.0, 200.0, 365.0],
        74: [2.0, 2.5, 200.0, 365.0],
        75: [2.5, 3.0, 200.0, 365.0],
        76: [3.0, 3.5, 200.0, 365.0],
        77: [3.5, 4.0, 200.0, 365.0],
    }

    rsamps = random.choice(balls, size=nselect)
    radius = np.zeros(nselect)
    period = np.zeros(nselect)
    for i, samp in enumerate(rsamps):
        rl, rh, pl, ph = ball_lookup[samp]
        radius[i] = random.uniform(low=rl, high=rh)
        period[i] = random.uniform(low=pl, high=ph)

    return radius, period


def Petigura18_select(nselect=1):
    # create a pot for pedigura numbers (balls)
    balls = np.array([])

    p1 = np.zeros(2) + 1
    p2 = np.zeros(8) + 2
    p3 = np.zeros(21) + 3
    p4 = np.zeros(8) + 4
    p5 = np.zeros(24) + 5
    p6 = np.zeros(52) + 6
    p7 = np.zeros(77) + 7
    p8 = np.zeros(5) + 8
    p9 = np.zeros(26) + 9
    p10 = np.zeros(24) + 10
    p11 = np.zeros(145) + 11
    p12 = np.zeros(259) + 12
    p13 = np.zeros(5) + 13
    p14 = np.zeros(12) + 14
    p15 = np.zeros(18) + 15
    p16 = np.zeros(17) + 16
    p17 = np.zeros(38) + 17
    p18 = np.zeros(168) + 18
    p19 = np.zeros(12) + 19
    p20 = np.zeros(8) + 20
    p21 = np.zeros(25) + 21
    p22 = np.zeros(56) + 22
    p23 = np.zeros(53) + 23
    p24 = np.zeros(78) + 24
    p25 = np.zeros(84) + 25
    p26 = np.zeros(78) + 26
    p27 = np.zeros(6) + 27
    p28 = np.zeros(8) + 28
    p29 = np.zeros(94) + 29
    p30 = np.zeros(180) + 30
    p31 = np.zeros(185) + 31
    p32 = np.zeros(258) + 32
    p33 = np.zeros(275) + 33
    p34 = np.zeros(312) + 34
    p35 = np.zeros(225) + 35
    p36 = np.zeros(8) + 36
    p37 = np.zeros(77) + 37
    p38 = np.zeros(138) + 38
    p39 = np.zeros(423) + 39
    p40 = np.zeros(497) + 40
    p41 = np.zeros(667) + 41
    p42 = np.zeros(475) + 42
    p43 = np.zeros(270) + 43
    p44 = np.zeros(147) + 44
    p45 = np.zeros(8) + 45
    p46 = np.zeros(34) + 46
    p47 = np.zeros(125) + 47
    p48 = np.zeros(202) + 48
    p49 = np.zeros(279) + 49
    p50 = np.zeros(261) + 50
    p51 = np.zeros(251) + 51
    p52 = np.zeros(186) + 52
    p53 = np.zeros(360) + 53
    p54 = np.zeros(393) + 54
    p55 = np.zeros(12) + 55
    p56 = np.zeros(36) + 56
    p57 = np.zeros(141) + 57
    p58 = np.zeros(263) + 58
    p59 = np.zeros(450) + 59
    p60 = np.zeros(350) + 60
    p61 = np.zeros(287) + 61
    p62 = np.zeros(249) + 62
    p63 = np.zeros(12) + 63
    p64 = np.zeros(52) + 64
    p65 = np.zeros(128) + 65
    p66 = np.zeros(315) + 66
    p67 = np.zeros(205) + 67
    p68 = np.zeros(447) + 68
    p69 = np.zeros(8) + 69
    p70 = np.zeros(50) + 70

    balls = np.r_[
        balls,
        p1,
        p2,
        p3,
        p4,
        p5,
        p6,
        p7,
        p8,
        p9,
        p10,
        p11,
        p12,
        p13,
        p14,
        p15,
        p16,
        p17,
        p18,
        p19,
        p20,
        p21,
        p22,
        p23,
        p24,
        p25,
        p26,
        p27,
        p28,
        p29,
        p30,
        p31,
        p32,
        p33,
        p34,
        p35,
        p36,
        p37,
        p38,
        p39,
        p40,
        p41,
        p42,
        p43,
        p44,
        p45,
        p46,
        p47,
        p48,
        p49,
        p50,
        p51,
        p52,
        p53,
        p54,
        p55,
        p56,
        p57,
        p58,
        p59,
        p60,
        p61,
        p62,
        p63,
        p64,
        p65,
        p66,
        p67,
        p68,
        p69,
        p70,
    ]

    ball_lookup = {
        0: [0.0, 0.0, 0.0, 0.0],
        1: [11.31, 16.00, 1.00, 1.78],
        2: [11.31, 16.00, 1.78, 3.16],
        3: [11.31, 16.00, 3.16, 5.62],
        4: [11.31, 16.00, 5.62, 10.00],
        5: [11.31, 16.00, 31.62, 56.23],
        6: [11.31, 16.00, 100.00, 177.83],
        7: [11.31, 16.00, 177.83, 316.23],
        8: [8.00, 11.31, 3.16, 5.62],
        9: [8.00, 11.31, 17.78, 31.62],
        10: [8.00, 11.31, 31.62, 56.23],
        11: [8.00, 11.31, 100.00, 177.83],
        12: [8.00, 11.31, 177.83, 316.23],
        13: [5.66, 8.00, 3.16, 5.62],
        14: [5.66, 8.00, 5.62, 10.00],
        15: [5.66, 8.00, 10.00, 17.78],
        16: [5.66, 8.00, 17.78, 31.62],
        17: [5.66, 8.00, 31.62, 56.23],
        18: [5.66, 8.00, 177.83, 316.23],
        19: [4.00, 5.66, 3.16, 5.62],
        20: [4.00, 5.66, 5.62, 10.00],
        21: [4.00, 5.66, 10.00, 17.78],
        22: [4.00, 5.66, 17.78, 31.62],
        23: [4.00, 5.66, 31.62, 56.23],
        24: [4.00, 5.66, 56.23, 100.00],
        25: [4.00, 5.66, 100.00, 177.83],
        26: [4.00, 5.66, 177.83, 316.23],
        27: [2.83, 4.00, 1.78, 3.16],
        28: [2.83, 4.00, 3.16, 5.62],
        29: [2.83, 4.00, 5.62, 10.00],
        30: [2.83, 4.00, 10.00, 17.78],
        31: [2.83, 4.00, 17.78, 31.62],
        32: [2.83, 4.00, 31.62, 56.23],
        33: [2.83, 4.00, 56.23, 100.00],
        34: [2.83, 4.00, 100.00, 177.83],
        35: [2.83, 4.00, 177.83, 316.23],
        36: [2.00, 2.83, 1.78, 3.16],
        37: [2.00, 2.83, 3.16, 5.62],
        38: [2.00, 2.83, 5.62, 10.00],
        39: [2.00, 2.83, 10.00, 17.78],
        40: [2.00, 2.83, 17.78, 31.62],
        41: [2.00, 2.83, 31.62, 56.23],
        42: [2.00, 2.83, 56.23, 100.00],
        43: [2.00, 2.83, 100.00, 177.83],
        44: [2.00, 2.83, 177.83, 316.23],
        45: [1.41, 2.00, 1.00, 1.78],
        46: [1.41, 2.00, 1.78, 3.16],
        47: [1.41, 2.00, 3.16, 5.62],
        48: [1.41, 2.00, 5.62, 10.00],
        49: [1.41, 2.00, 10.00, 17.78],
        50: [1.41, 2.00, 17.78, 31.62],
        51: [1.41, 2.00, 31.62, 56.23],
        52: [1.41, 2.00, 56.23, 100.00],
        53: [1.41, 2.00, 100.00, 177.83],
        54: [1.41, 2.00, 177.83, 316.23],
        55: [1.00, 1.41, 1.00, 1.78],
        56: [1.00, 1.41, 1.78, 3.16],
        57: [1.00, 1.41, 3.16, 5.62],
        58: [1.00, 1.41, 5.62, 10.00],
        59: [1.00, 1.41, 10.00, 17.78],
        60: [1.00, 1.41, 17.78, 31.62],
        61: [1.00, 1.41, 31.62, 56.23],
        62: [1.00, 1.41, 56.23, 100.00],
        63: [0.71, 1.00, 1.00, 1.78],
        64: [0.71, 1.00, 1.78, 3.16],
        65: [0.71, 1.00, 3.16, 5.62],
        66: [0.71, 1.00, 5.62, 10.00],
        67: [0.71, 1.00, 10.00, 17.78],
        68: [0.71, 1.00, 17.78, 31.62],
        69: [0.50, 0.71, 1.00, 1.78],
        70: [0.50, 0.71, 1.78, 3.16],
    }

    rsamps = random.choice(balls, size=nselect)
    radius = np.zeros(nselect)
    period = np.zeros(nselect)
    for i, samp in enumerate(rsamps):
        rl, rh, pl, ph = ball_lookup[samp]

        if samp in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            # check for giant planets
            # if a giant planet than draw power law
            radius[i] = rndm(8, 16, -1.7)
        else:
            radius[i] = random.uniform(low=rl, high=rh)

        period[i] = random.uniform(low=pl, high=ph)

    return radius, period


def per2ars(per, mstar, rstar):
    per_SI = per * 86400.0
    mass_SI = mstar * msun
    a3 = per_SI ** 2 * G * mass_SI / (4 * np.pi ** 2)
    return a3 ** (1.0 / 3.0) / (rstar * rsun)


def get_duration(per, ars, cosi=0.0, b=0, rprs=0.0):
    """
    returns the transit duration in days
    """
    part1 = per / np.pi
    part2 = 1.0 / ars
    part3 = np.sqrt((1 + rprs) ** 2 - b ** 2)
    part4 = np.sqrt(1 - cosi ** 2)
    duration = part1 * np.arcsin(part2 * part3 / part4)

    return duration


def get_transit_depth(Prad, rstar_solar):
    """
    returns transit depth in ppm
    """
    tdep = (Prad * 0.009155 / rstar_solar) ** 2 * 1.0e6  # ppm
    return tdep


def get_rprs(Prad, rstar_solar):
    return (Prad * 0.009155) / rstar_solar


def make_allplanets_df_vec_extrap(df, starid_zp):
    # lets refector the above code to make it array operations
    totalRows = df.loc[:, "Nplanets"].sum()

    df.loc[:, "planetRadius"] = pd.Series()
    df.loc[:, "planetPeriod"] = pd.Series()
    df.loc[:, "starID"] = pd.Series()

    radper_m = Dressing15_select_extrap(totalRows)
    radper_fgk = Petigura18_select(totalRows)

    # we need an array of indices
    rowIdx = np.repeat(np.arange(df.shape[0]), np.array(df.Nplanets.values))

    newdf = df.iloc[rowIdx]
    newdf.loc[:, "starID"] = rowIdx + starid_zp

    newdf.loc[:, "planetRadius"] = np.where(
        newdf.isMdwarf, radper_m[0], radper_fgk[0]
    )
    newdf.loc[:, "planetPeriod"] = np.where(
        newdf.isMdwarf, radper_m[1], radper_fgk[1]
    )
    newdf.set_index(np.arange(newdf.shape[0]), inplace=True)

    return newdf, newdf.starID.iloc[-1]


def kepler_noise_1h(kepmag):
    # 1 hour CDPP
    # these numbers are from the Q14 measured rmscdpp
    mag_level, noise_level = np.array(
        [
            [0.0, 20.0],
            [3.0, 20.0],
            [6.0, 20.0],
            [8.0, 20.0],
            [9.00995575221239, 20.000000000000057],
            [9.120575221238939, 22.523364485981347],
            [9.253318584070797, 23.925233644859844],
            [9.380530973451327, 25.607476635514047],
            [9.59070796460177, 27.570093457943983],
            [9.773230088495575, 28.41121495327107],
            [9.972345132743364, 28.691588785046775],
            [10.143805309734514, 29.252336448598186],
            [10.326327433628318, 28.97196261682248],
            [10.525442477876107, 28.97196261682248],
            [10.719026548672566, 28.691588785046775],
            [10.857300884955752, 28.97196261682248],
            [11.045353982300885, 28.97196261682248],
            [11.27212389380531, 29.813084112149596],
            [11.48783185840708, 31.214953271028065],
            [11.692477876106196, 32.05607476635518],
            [11.819690265486726, 32.89719626168227],
            [11.996681415929203, 34.57943925233647],
            [12.13495575221239, 35.420560747663586],
            [12.267699115044248, 36.822429906542084],
            [12.411504424778762, 37.943925233644904],
            [12.56637168141593, 39.62616822429911],
            [12.71570796460177, 41.028037383177605],
            [12.876106194690266, 43.27102803738322],
            [13.069690265486727, 45.794392523364536],
            [13.252212389380531, 48.03738317757015],
            [13.4070796460177, 51.12149532710285],
            [13.561946902654867, 54.20560747663555],
            [13.733407079646017, 58.130841121495365],
            [13.83849557522124, 60.37383177570098],
            [13.971238938053098, 64.2990654205608],
            [14.065265486725664, 67.6635514018692],
            [14.153761061946902, 70.74766355140193],
            [14.231194690265488, 73.55140186915892],
            [14.308628318584072, 76.35514018691595],
            [14.386061946902656, 79.71962616822435],
            [14.446902654867257, 82.24299065420567],
            [14.513274336283185, 85.32710280373837],
            [14.596238938053098, 89.53271028037389],
            [14.690265486725664, 94.01869158878509],
            [14.767699115044248, 97.66355140186923],
            [14.823008849557523, 101.02803738317763],
            [14.883849557522126, 104.95327102803745],
            [14.96128318584071, 109.43925233644865],
            [15.011061946902656, 112.52336448598138],
        ]
    ).T
    mag_interp = interp1d(
        mag_level, noise_level, kind="linear", fill_value="extrapolate"
    )
    return mag_interp(kepmag) * np.sqrt(6.5)

def kepler_noise_1h_quiet(kepmag):
    # 1 hour CDPP
    # this is calculated from the rrmscdpp06p0
    mag_level, noise_level = np.array(
        [
            [0.0, 20.0],
            [3.0, 20.0],
            [6.0, 20.0],
            [8.0, 20.0],
            [9.00995575221239, 20.000000000000057],
            [9.2, 21.2625],
            [9.299999999999999, 20],
            [9.399999999999999, 14.389000000000001],
            [9.499999999999998, 24.667499999999997],
            [9.599999999999998, 24.392500000000005],
            [9.699999999999998, 26.223],
            [9.799999999999997, 19.779],
            [9.899999999999997, 14.007],
            [9.999999999999996, 17.862000000000005],
            [10.099999999999996, 20.965],
            [10.299999999999995, 20.464],
            [10.399999999999995, 19.271],
            [10.499999999999995, 16.5505],
            [10.599999999999994, 21.195999999999998],
            [10.699999999999994, 26.0565],
            [10.799999999999994, 27.654],
            [10.899999999999993, 25.377],
            [10.999999999999993, 22.171],
            [11.099999999999993, 24.851],
            [11.199999999999992, 24.87],
            [11.299999999999992, 27.1965],
            [11.399999999999991, 25.774],
            [11.499999999999991, 27.665],
            [11.59999999999999, 30.0305],
            [11.69999999999999, 31.01],
            [11.79999999999999, 32.178],
            [11.89999999999999, 31.628],
            [11.99999999999999, 32.558],
            [12.099999999999989, 35.0385],
            [12.199999999999989, 35.259],
            [12.299999999999988, 36.119],
            [12.399999999999988, 37.184],
            [12.499999999999988, 39.861999999999995],
            [12.599999999999987, 41.931000000000004],
            [12.699999999999987, 42.528],
            [12.799999999999986, 43.259],
            [12.899999999999986, 45.439],
            [12.999999999999986, 49.3505],
            [13.099999999999985, 50.164],
            [13.199999999999985, 53.51300000000001],
            [13.299999999999985, 55.575],
            [13.399999999999984, 57.218999999999994],
            [13.499999999999984, 60.161500000000004],
            [13.599999999999984, 62.68],
            [13.699999999999983, 65.464],
            [13.799999999999983, 70.37],
            [13.899999999999983, 73.724],
            [13.999999999999982, 77.017],
            [14.099999999999982, 81.047],
            [14.199999999999982, 85.068],
            [14.299999999999981, 89.715],
            [14.39999999999998, 95.31],
            [14.49999999999998, 101.193],
            [14.59999999999998, 106.978],
            [14.69999999999998, 112.5995],
            [14.79999999999998, 118.04700000000001],
            [14.899999999999979, 125.4615],
            [14.999999999999979, 133.9125],
            [15.099999999999978, 141.15500000000003],
            [15.199999999999978, 149.125],
            [15.299999999999978, 159.1295],
            [15.399999999999977, 168.91],
            [15.499999999999977, 179.018],
            [15.599999999999977, 192.773],
            [15.699999999999976, 202.986],
            [15.799999999999976, 218.581],
            [15.899999999999975, 234.59900000000002],
            [15.999999999999975, 245.80700000000002],
            [16.099999999999973, 287.57599999999996],
            [16.199999999999974, 282.94399999999996],
            [16.299999999999976, 270.305],
            [16.399999999999974, 321.54200000000003],
            [16.49999999999997, 359.365],
            [16.599999999999973, 349.54400000000015],
            [16.699999999999974, 417.082],
            [16.799999999999972, 425.254],
            [16.89999999999997, 419.8280000000001],
            [17.099999999999973, 434.58],
        ]
    ).T
    mag_interp = interp1d(
        mag_level, noise_level, kind="linear", fill_value="extrapolate"
    )
    return mag_interp(kepmag) * np.sqrt(6.)


def make_allplanets_df_vec_extrap_kepler(df, starid_zp, ocrMeasurement='bryson'):
    totalRows = df.loc[:, "Nplanets"].sum()

    df.loc[:, "planetRadius"] = pd.Series()
    df.loc[:, "planetPeriod"] = pd.Series()
    df.loc[:, "starID"] = pd.Series()

    radper_m = Dressing15_select_extrap(totalRows)
    radper_fgk = Bryson_select(totalRows)

    # we need an array of indices
    rowIdx = np.repeat(np.arange(df.shape[0]), np.array(df.Nplanets.values))

    newdf = df.iloc[rowIdx]
    newdf.loc[:, "starID"] = rowIdx + starid_zp

    newdf.loc[:, "planetRadius"] = np.where(
        newdf.isMdwarf, radper_m[0], radper_fgk[0]
    )
    newdf.loc[:, "planetPeriod"] = np.where(
        newdf.isMdwarf, radper_m[1], radper_fgk[1]
    )
    newdf.set_index(np.arange(newdf.shape[0]), inplace=True)

    return newdf, newdf.starID.iloc[-1]


def Bryson_select(nselect=1, ocrMeasurement='bryson'):
    balls = np.array([])

    if ocrMeasurement == 'bryson':
        fn_occ = "../data/bryson/occurrenceGrid_860_bryson.npy"
    elif ocrMeasurement == 'burke':
        fn_occ = "../data/bryson/occurrenceGrid_860_burke.npy"
    elif ocrMeasurement == 'LUVOIR':
        # simulate the LUVOIR eta-earth planets
        radius = np.zeros(nselect)
        period = np.zeros(nselect)
        radius = random.uniform(low=0.8, high=1.4, size=nselect)
        period = random.uniform(low=338, high=778, size=nselect)
        return radius, period
    fn_p = "../data/bryson/occurrencePeriod_860.npy"
    fn_r = "../data/bryson/occurrenceRadius_860.npy"
    ocrGrid = np.load(fn_occ)
    rp1D = np.load(fn_r)
    period1D = np.load(fn_p)

    # use a 100 x 100 grid
    occBalls = np.round(ocrGrid.flatten() * 1.0e6)
    for i in range(occBalls.shape[0]):
        balls = np.r_[balls, np.zeros(int(occBalls[i])) + i]

    ball_lookup = {}
    dPeriod = period1D[1] - period1D[0]
    dRadius = rp1D[1] - rp1D[0]
    for i in range(200):
        for j in range(100):
            ball_lookup[i * 100 + j] = [
                rp1D[j] - dRadius / 2,
                rp1D[j] + dRadius / 2,
                period1D[i] - dPeriod / 2,
                period1D[i] + dPeriod / 2,
            ]

    rsamps = random.choice(balls, size=nselect)
    radius = np.zeros(nselect)
    period = np.zeros(nselect)
    for i, samp in enumerate(rsamps):
        rl, rh, pl, ph = ball_lookup[samp]

        radius[i] = random.uniform(low=rl, high=rh)
        period[i] = random.uniform(low=pl, high=ph)

    return radius, period
