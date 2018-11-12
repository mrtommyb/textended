from astropy.coordinates import SkyCoord
from tessmaps import get_time_on_silicon as gts

coords = SkyCoord([150,150], [-50,50], unit='deg')

df = gts.get_time_on_silicon(coords)