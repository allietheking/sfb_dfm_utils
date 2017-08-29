import os
import numpy as np
import xarray as xr

import stompy.model.delft.io as dio
from . import dredge_grid

DAY=np.timedelta64(86400,'s') # useful for adjusting times

def add_sfbay_freshwater(run_base_dir,
                         run_start,run_stop,ref_date,
                         adjusted_pli_fn,
                         freshwater_dir,
                         grid,dredge_depth,
                         old_bc_fn,
                         all_flows_unit=False):
    """
    Add freshwater flows from sfbay_freshwater git submodule.
    run_base_dir: location of DFM input files
    run_start,run_stop: target period for run, as np.datetime64
    ref_date: DFM reference date, as np.datetime64[D]
    adjusted_pli_fn: path to pli file to override source locations
    freshwater_dir: path to sfbay_freshwater git submodule
    grid: UnstructuredGrid instance to be modified at input locations
    old_bc_fn: path to old-style forcing input file
    """
    adjusted_features=dio.read_pli(adjusted_pli_fn)
    # Add the freshwater flows - could come from erddap, but use github submodule
    # for better control on version

    # create a pair of bc and pli files, each including all the sources.
    # exact placement will
    # be done by hand in the GUI

    full_flows_ds = xr.open_dataset(os.path.join(freshwater_dir, 'outputs', 'sfbay_freshwater.nc'))

    # period of the full dataset which will be include for this run
    sel=(full_flows_ds.time > run_start - 5*DAY) & (full_flows_ds.time < run_stop + 5*DAY)
    flows_ds = full_flows_ds.isel(time=sel)

    for stni in range(len(flows_ds.station)):
        stn_ds=flows_ds.isel(station=stni)

        src_name=stn_ds.station.item() # kind of a pain to get scalar values back out...

        # At least through the GUI, pli files must have more than one node.
        # Don't get too big for our britches, just stick a second node 50m east
        # if the incoming data is a point, but check for manually set locations
        # in adjusted_features
        if 1: #-- Write a PLI file
            feat=(src_name,
                  np.array( [[stn_ds.utm_x,stn_ds.utm_y],
                             [stn_ds.utm_x + 50.0,stn_ds.utm_y]] ))
            # Scan adjusted features for a match to use instead
            for adj_feat in adjusted_features:
                if adj_feat[0] == src_name:
                    feat=adj_feat
                    break
            # Write copies for flow, salinity and temperatures
            for suffix in ['_flow','_salt','_temp']:
                # function to add suffix
                feat_suffix=dio.add_suffix_to_feature(feat,suffix)
                pli_fn=os.path.join(run_base_dir,"%s%s.pli"%(src_name,suffix))
                dio.write_pli(pli_fn,[feat_suffix])

            dredge_grid.dredge_boundary(grid,feat[1],dredge_depth)

        if 1: #-- Write the time series and stanza in FlowFM_bnd.ext
            df=stn_ds.to_dataframe().reset_index()
            df['elapsed_minutes']=(df.time.values - ref_date)/np.timedelta64(60,'s')
            df['salinity']=0*df.flow_cms
            df['temperature']=20+0*df.flow_cms

            if all_flows_unit:
                df['flow_cms']=1.0+0*df.flow_cms
                
            for quantity,suffix in [ ('dischargebnd','_flow'),
                                     ('salinitybnd','_salt'),
                                     ('temperaturebnd','_temp') ]:
                lines=['QUANTITY=%s'%quantity,
                       'FILENAME=%s%s.pli'%(src_name,suffix),
                       'FILETYPE=9',
                       'METHOD=3',
                       'OPERAND=O',
                       ""]
                with open(old_bc_fn,'at') as fp:
                    fp.write("\n".join(lines))

                # read the pli back to know how to name the per-node timeseries
                feats=dio.read_pli(os.path.join(run_base_dir,
                                                "%s%s.pli"%(src_name,suffix)))
                feat=feats[0] # just one polyline in the file

                if len(feat)==3:
                    node_names=feat[2]
                else:
                    node_names=[""]*len(feat[1])
                    
                for node_idx,node_name in enumerate(node_names):
                    # if no node names are known, create the default name of <feature name>_0001
                    if not node_name:
                        node_name="%s%s_%04d"%(src_name,suffix,1+node_idx)

                    tim_fn=os.path.join(run_base_dir,node_name+".tim")
                    
                    columns=['elapsed_minutes']
                    if quantity=='dischargebnd':
                        columns.append('flow_cms')
                    elif quantity=='salinitybnd':
                        columns.append('salinity')
                    elif quantity=='temperaturebnd':
                        columns.append('temperature')
                    
                    df.to_csv(tim_fn, sep=' ', index=False, header=False, columns=columns)

