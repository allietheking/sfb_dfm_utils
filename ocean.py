import os
import logging
log=logging.getLogger('sfb_dfm_utils')

import numpy as np
import xarray as xr

from stompy import (utils, filters)
from stompy.io.local import noaa_coops
import stompy.model.delft.io as dio

# fill any missing data via linear interpolation
def fill_data(da):
    valid=np.isfinite(da.values)
    valid_times=da.time[valid]
    dt=np.median(np.diff(da.time))
    dt_gap = np.diff(da.time[valid]).max()
    if dt_gap > dt:
        log.warning("%s: gaps up to %.1f minutes"%(da.name, dt_gap/np.timedelta64(60,'s')))
    da.values = utils.fill_invalid(da.values)    

def add_ocean(run_base_dir,
              run_start,run_stop,ref_date,
              static_dir,
              grid,old_bc_fn,
              all_flows_unit=False):
    """
    Ocean:
    Water level data from station 46214 (apparently from Yi Chao's ROMS?)
      no spatial variation
    Maybe salinity from Yi Chao ROMS?  That's what the thesis says, but the
    actual inputs look like constant 33

    Ocean BCs from Point Reyes
    """
    # get a few extra days of data to allow for transients in the low pass filter.
    pad_time=np.timedelta64(5,'D')
    
    ptreyes_raw_fn=os.path.join(run_base_dir,'ptreyes-raw.nc')
    if 1:
        if not os.path.exists(ptreyes_raw_fn):
            ptreyes_raw=noaa_coops.coops_dataset("9415020",
                                                 run_start-pad_time,run_stop+pad_time,
                                                 ["water_level","water_temperature"],
                                                 days_per_request=30)

            ptreyes_raw.to_netcdf(ptreyes_raw_fn,engine='scipy')

    if 1:
        # Clean that up, fabricate salinity
        ptreyes=xr.open_dataset(ptreyes_raw_fn).isel(station=0)

        water_level=utils.fill_tidal_data(ptreyes.water_level)

        if 0: # FIR filter, has to be shorter to avoid attenuation
            # And lowpass at 1 hour to get rid of wave energy
            # with the fir filter, that's about a 2% amplitude loss at 6h.
            winsize_lp=int( np.timedelta64(2,'h') / np.median(np.diff(water_level[:].time)) ) 
            water_level[:] = filters.lowpass_fir(water_level[:].values,winsize_lp)
        else: # IIR butterworth.  Nicer, with minor artifacts at ends
            # 3 hours, defaults to 4th order.
            water_level[:] = filters.lowpass(water_level[:].values,
                                             utils.to_dnum(water_level.time),
                                             cutoff=3./24)
                                             
                                             
        fill_data(ptreyes.water_temperature)
        water_temp=ptreyes.water_temperature

        if all_flows_unit:
            print("-=-=-=- USING 35 PPT WHILE TESTING! -=-=-=-")
            salinity=35 + 0*ptreyes.water_temperature
        else:
            salinity=33 + 0*ptreyes.water_temperature
        salinity.name='salinity'
            
    if 1: # Write it all out
        # Add a stanza to FlowFMold_bnd.ext:
        src_name='Sea'
        
        src_feat=dio.read_pli(os.path.join(static_dir,'%s.pli'%src_name))[0]
        
        forcing_data=[('waterlevelbnd',water_level,'_ssh'),
                      ('salinitybnd',salinity,'_salt'),
                      ('temperaturebnd',water_temp,'_temp')]

        for quant,da,suffix in forcing_data:
            with open(old_bc_fn,'at') as fp:
                lines=["QUANTITY=%s"%quant,
                       "FILENAME=%s%s.pli"%(src_name,suffix),
                       "FILETYPE=9",
                       "METHOD=3",
                       "OPERAND=O",
                       ""]
                fp.write("\n".join(lines))

            feat_suffix=dio.add_suffix_to_feature(src_feat,suffix)
            dio.write_pli(os.path.join(run_base_dir,'%s%s.pli'%(src_name,suffix)),
                          [feat_suffix])

            # Write the data:
            columns=['elapsed_minutes',da.name]

            df=da.to_dataframe().reset_index()
            df['elapsed_minutes']=(df.time.values - ref_date)/np.timedelta64(60,'s')

            if len(feat_suffix)==3:
                node_names=feat_suffix[2]
            else:
                node_names=[""]*len(feat_suffix[1])

            for node_idx,node_name in enumerate(node_names):
                # if no node names are known, create the default name of <feature name>_0001
                if not node_name:
                    node_name="%s%s_%04d"%(src_name,suffix,1+node_idx)

                tim_fn=os.path.join(run_base_dir,node_name+".tim")
                df.to_csv(tim_fn, sep=' ', index=False, header=False, columns=columns)
