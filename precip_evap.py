"""
A first cut at direct precipitation and evaporation.

This depends on '/opt/data/cimis/union_city-hourly-2001-2016.nc'
which is currently downloaded through 2017-01-08 on hpc.
"""
import xarray as xr
from stompy import utils

# Last SUNTANS run had used NARR
# it's way way coarse.  Seems better to use an in-Bay climatology
# than to use NARR.

##

def load_cimis():
    union_city=xr.open_dataset('/opt/data/cimis/union_city-hourly-2001-2016.nc')

    # https://cals.arizona.edu/azmet/et1.htm
    # which says cool period, divide ETO by 0.7 to get pan evaporation,
    # warm period, divide by 0.6.

    temps=utils.fill_invalid(union_city.HlyAirTmp.values)
    temp_zscore=((temps-temps.mean()) / temps.std()).clip(-1,1)
    # score of 1 means warm temperature
    factors=np.interp(temp_zscore,
                      [-1,1],[0.7,0.6])
    union_city['HlyEvap']=union_city.HlyEto/factors

    return union_city

# Data are in mm/hour
# relevant fields are union_city.time, union_city.HlyEvap, and union_city.HlyPrecip


def add_cimis_evap_precip(run_base_dir,run_start,run_stop,old_bc_fn):
