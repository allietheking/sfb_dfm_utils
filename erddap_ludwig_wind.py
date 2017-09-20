import os
import logging

import numpy as np
import xarray as xr

import stompy.model.delft.io as dio

log=logging.getLogger('sfb_dfm_utils')

DAY=np.timedelta64(86400,'s') # useful for adjusting times

def add_erddap_ludwig_wind(run_base_dir,run_start,run_stop,old_bc_fn):
    """
    fetch wind data, write to format supported by DFM, and append wind
    forcing stanzas to old-style DFM boundary forcing file.

    Wind data are fetched from ERDDAP, and written to DFM format
    
    run_base_dir: path to the run
    run_start,run_stop: target period of the run as datetime64
    old_bc_fn: path to the old-style boundary forcing file
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
            return

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

