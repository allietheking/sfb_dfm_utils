import os
import logging

import numpy as np
import xarray as xr

import stompy.model.delft.io as dio
from stompy.model.delft import dfm_grid

log=logging.getLogger('sfb_dfm_utils')

DAY=np.timedelta64(86400,'s') # useful for adjusting times

def add_erddap_ludwig_wind(run_base_dir,run_start,run_stop,old_bc_fn,fallback=None):
    """
    fetch wind data, write to format supported by DFM, and append wind
    forcing stanzas to old-style DFM boundary forcing file.

    Wind data are fetched from ERDDAP, and written to DFM format
    
    run_base_dir: path to the run
    run_start,run_stop: target period of the run as datetime64
    old_bc_fn: path to the old-style boundary forcing file

    fallback: None, or [u,v] for constant in space/time when no ludwig available

    returns True is data was found, otherwise return False
    """
    target_filename_base=os.path.join(run_base_dir,"wind") # will have amu, amv appended

    wind_u_fn=target_filename_base+".amu"
    wind_v_fn=target_filename_base+".amv"
    if (os.path.exists(wind_u_fn) and
        os.path.exists(wind_v_fn)):
        log.info('Wind files already exist')
    else:
        data_start=run_start-1*DAY
        data_stop=run_stop+1*DAY

        url='http://sfbaynutrients.sfei.org/erddap/griddap/wind_ludwig_20170621'
        ds=xr.open_dataset(url)

        # somewhat arbitrary cutoff of at least 4 per day
        min_records=(data_stop-data_start)/DAY * 4
        avail_records=dio.dataset_to_dfm_wind(ds,data_start,data_stop,target_filename_base,
                                              min_records=min_records,
                                              extra_header="# downloaded from %s"%url)
        if avail_records<min_records:
            log.warning("Wind data not available for %s -- %s"%(run_start.astype('M8[D]'),
                                                                run_stop.astype('M8[D]')))
            return False

    # and add wind to the boundary forcing
    wind_stanza=["QUANTITY=windx",
                 "FILENAME=wind.amu",
                 "FILETYPE=4",
                 "METHOD=2",
                 "OPERAND=O",
                 "",
                 "QUANTITY=windy",
                 "FILENAME=wind.amv",
                 "FILETYPE=4",
                 "METHOD=2",
                 "OPERAND=O",
                 "\n"]
    with open(old_bc_fn,'at') as fp:
        fp.write("\n".join(wind_stanza))
    return True


def add_constant_wind(run_base_dir,mdu,wind,run_start,run_stop):
    grid_fn=os.path.join(run_base_dir,mdu['geometry','NetFile'])
    
    g=dfm_grid.DFMGrid(grid_fn)

    # manufacture a constant in time, constant in space wind field
    ds=xr.Dataset()

    xxyy=g.bounds()
    pad=0.1*(xxyy[1]-xxyy[0])

    x=np.linspace(xxyy[0]-pad,xxyy[1]+pad,2)
    y=np.linspace(xxyy[2]-pad,xxyy[3]+pad,3)

    DAY=np.timedelta64(1,'D')
    data_start=run_start-1*DAY
    data_stop=run_stop+1*DAY

    t=np.arange(data_start,data_stop,np.timedelta64(3,'h'))

    ds['x']=('x',),x
    ds['y']=('y',),y
    ds['time']=('time',),t

    ds['wind_u']=('time','y','x'),wind[0]*np.ones( (len(t),len(y),len(x)) )
    ds['wind_v']=('time','y','x'),wind[1]*np.ones( (len(t),len(y),len(x)) )

    count=dio.dataset_to_dfm_wind(ds,data_start,data_stop,
                                  target_filename_base=os.path.join(run_base_dir,"const_wind"),
                                  extra_header="# constant %.2f %.2f wind"%(wind[0],wind[1]))
    assert count>0

    # and add wind to the boundary forcing
    wind_stanza=["QUANTITY=windx",
                 "FILENAME=const_wind.amu",
                 "FILETYPE=4",
                 "METHOD=2",
                 "OPERAND=O",
                 "",
                 "QUANTITY=windy",
                 "FILENAME=const_wind.amv",
                 "FILETYPE=4",
                 "METHOD=2",
                 "OPERAND=O",
                 "\n"]
    old_bc_fn=os.path.join(run_base_dir,mdu['external forcing','ExtForceFile'])
    
    with open(old_bc_fn,'at') as fp:
        fp.write("\n".join(wind_stanza))
    return True
