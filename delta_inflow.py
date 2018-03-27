import os
import numpy as np
import xarray as xr

import stompy.model.delft.io as dio
from stompy import utils
from stompy.io.local import usgs_nwis

from . import (dredge_grid,common)

# copied Silvia's pli files to inputs-static
# even though there are 14 of these, one per node of the sea boundary,
# they all appear to have the same data, even the tidal height time series.
# Sea_temp: probably grabbed from Point Reyes?
# Sea_sal: constant 33
# Sea_0001.pli - 

def add_delta_inflow(run_base_dir,
                     run_start,run_stop,ref_date,
                     static_dir,
                     grid,dredge_depth,
                     old_bc_fn,
                     all_flows_unit=False):
    """
    Fetch river USGS river flows, add to FlowFM_bnd.ext:
    Per Silvia's Thesis:
    Jersey: Discharge boundary affected by tides, discharge and temperature taken
    from USGS 11337190 SAN JOAQUIN R A JERSEY POINT, 0 salinity
    (Note that Dutch Slough should probably be added in here)
    Rio Vista: 11455420 SACRAMENTO A RIO VISTA, temperature from DWR station RIV.
    0 salinity.

    run_base_dir: location of the DFM inputs
    run_start,run_stop: target period for therun
    statiC_dir: path to static assets, specifically Jersey.pli and RioVista.pli
    grid: UnstructuredGrid instance, to be modified at inflow locations
    old_bc_fn: path to old-style boundary forcing file
    all_flows_unit: if True, override all flows to be 1 m3 s-1 for model diagnostics
    """

    pad=np.timedelta64(3,'D')
    
    if 0:  # cache here.
        # Cache the original data from USGS, then clean it and write to DFM format
        jersey_raw_fn=os.path.join(run_base_dir,'jersey-raw.nc')
        if not os.path.exists(jersey_raw_fn):
            jersey_raw=usgs_nwis.nwis_dataset(station="11337190",
                                              start_date=run_start-pad,end_date=run_stop+pad,
                                              products=[60, # "Discharge, cubic feet per second"
                                                        10], # "Temperature, water, degrees Celsius"
                                              days_per_request=30)
            jersey_raw.to_netcdf(jersey_raw_fn,engine='scipy')
        else:
            jersey_raw=xr.open_dataset(jersey_raw_fn)

        rio_vista_raw_fn=os.path.join(run_base_dir,'rio_vista-raw.nc')
        if not os.path.exists(rio_vista_raw_fn):
            rio_vista_raw=usgs_nwis.nwis_dataset(station="11455420",
                                                 start_date=run_start-pad,end_date=run_stop+pad,
                                                 products=[60, # "Discharge, cubic feet per second"
                                                           10], # "Temperature, water, degrees Celsius"
                                                 days_per_request=30)
            rio_vista_raw.to_netcdf(rio_vista_raw_fn,engine='scipy')
        else:
            rio_vista_raw=xr.open_dataset(rio_vista_raw_fn)
    else:  # cache in nwis code
        jersey_raw=usgs_nwis.nwis_dataset(station="11337190",
                                          start_date=run_start-pad,end_date=run_stop+pad,
                                          products=[60, # "Discharge, cubic feet per second"
                                                    10], # "Temperature, water, degrees Celsius"
                                          days_per_request='M',
                                          cache_dir=common.cache_dir)

        rio_vista_raw=usgs_nwis.nwis_dataset(station="11455420",
                                             start_date=run_start-pad,end_date=run_stop+pad,
                                             products=[60, # "Discharge, cubic feet per second"
                                                       10], # "Temperature, water, degrees Celsius"
                                             days_per_request='M',
                                             cache_dir=common.cache_dir)

    if 1: # Clean and write it all out
        for src_name,source in [ ('Jersey',jersey_raw),
                                 ('RioVista',rio_vista_raw)]:
            src_feat=dio.read_pli(os.path.join(static_dir,'%s.pli'%src_name))[0]
            
            dredge_grid.dredge_boundary(grid,src_feat[1],dredge_depth)
            
            # Add stanzas to FlowFMold_bnd.ext:
            for quant,suffix in [('dischargebnd','_flow'),
                                 ('salinitybnd','_salt'),
                                 ('temperaturebnd','_temp')]:
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
                if quant=='dischargebnd':
                    da=source.stream_flow_mean_daily
                    da2=utils.fill_tidal_data(da)
                    if all_flows_unit:
                        da2.values[:]=1.0
                    else:
                        # convert ft3/s to m3/s
                        da2.values[:] *= 0.028316847 
                elif quant=='salinitybnd':
                    da2=source.stream_flow_mean_daily.copy(deep=True)
                    da2.values[:]=0.0
                elif quant=='temperaturebnd':
                    da=source.temperature_water
                    da2=utils.fill_tidal_data(da) # maybe safer to just interpolate?
                    if all_flows_unit:
                        da2.values[:]=20.0
                        
                df=da2.to_dataframe().reset_index()
                df['elapsed_minutes']=(df.time.values - ref_date)/np.timedelta64(60,'s')
                columns=['elapsed_minutes',da2.name]
                    
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
