"""
A first cut at direct precipitation and evaporation.

This depends on '/opt/data/cimis/union_city-hourly-2001-2016.nc'
which is currently downloaded through 2017-01-08 on hpc.
"""
import os
import datetime

import xarray as xr
import numpy as np
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

    union_city.time.values += np.timedelta64(8,'h')
    return union_city

# Data are in mm/hour
# relevant fields are union_city.time, union_city.HlyEvap, and union_city.HlyPrecip
# Times are adjusted from assumed PST to UTC


def mdu_time_range(mdu):
    t_ref=utils.to_dt64( datetime.datetime.strptime(mdu['time','RefDate'],'%Y%m%d') )

    if mdu['time','Tunit'].lower() == 'm':
        tunit=np.timedelta64(1,'m')
    else:
        raise Exception("TODO: allow other time units")
    
    t_start = t_ref+int(mdu['time','tstart'])*tunit
    t_stop = t_ref+int(mdu['time','tstop'])*tunit
    return t_ref,t_start,t_stop

def add_cimis_evap_precip(run_base_dir,mdu,scale_precip=1.0,scale_evap=1.0):
    data = load_cimis()

    pad=np.timedelta64(5,'D')

    t_ref,t_start,t_stop=mdu_time_range(mdu)

    old_bc_fn=os.path.join(run_base_dir,mdu['external forcing','ExtForceFile'])

    with open(old_bc_fn,'at') as fp:
        # some mentions of "rainfall", other times "rain"
        lines=["QUANTITY=rainfall",
               "FILENAME=precip_evap.tim",
               "FILETYPE=1", # uniform
               "METHOD=1", # ?? interpolate in space and time?
               "OPERAND=O",
               ""]
        fp.write("\n".join(lines))

    sel=((data.time>=t_start-pad)&(data.time<=t_stop+pad)).values
    minutes=(data.time.values[sel] - t_ref) / np.timedelta64(60,'s')

    # starting unit: mm/hr
    # What unit do they want? some mentions in the source of mm/hr, other times
    # mm/day.  I think they want mm/day.  Initial look at the rain values in the
    # output confirm that units are good.
    net_precip=24*(scale_precip*data.HlyPrecip.values[sel] - scale_evap*data.HlyEvap.values[sel])

    time_series = np.c_[minutes,net_precip]
    
    np.savetxt(os.path.join(run_base_dir,'precip_evap.tim'),
               time_series,
               fmt="%g")
        

        
