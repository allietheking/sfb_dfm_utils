"""
Utilities to download CA ROMS data to local cache, extract grids, 
and extract BC data.
"""

import time
import os
import glob
import logging

logger=logging.getLogger('ca_roms')

import numpy as np
import xarray as xr

from . import local_config

from stompy import utils, memoize

from stompy.grid import unstructured_grid
from stompy.spatial import wkb2shp
from stompy.plot import plot_utils, plot_wkb
from stompy.spatial import proj_utils, wkb2shp,field, linestring_utils
from stompy.model.delft import dfm_grid

from shapely import geometry
from shapely.ops import cascaded_union


cache_path = os.path.join(local_config.cache_path,'ca_roms')

utm2ll=proj_utils.mapper('EPSG:26910','WGS84')
ll2utm=proj_utils.mapper('WGS84','EPSG:26910')


def fetch_ca_roms(start,stop):
    """
    Download the 6-hourly outputs for the given period.
    start,stop: np.datetime64 
    returns a list of paths to local files falling in that period

    Pads the dates out by 12-36 h, as the data files are every 6h, staggered,
    and the stop date gets truncated 1 day
    """

    start=start-np.timedelta64(12,'h')
    stop=stop+np.timedelta64(36,'h')
    
    # Ways of getting the list of files:
    # http://west.rssoffice.com:8080/thredds/catalog/roms/CA3km-nowcast/CA/catalog.xml
    # That provides xml with file names, sizes, but no metadata about period simulated.

    assert os.path.exists(cache_path)
    
    local_files=[]
        
    # As a first step, construct the url by hand:
    for day in np.arange(start,stop,np.timedelta64(1,'D')):
        logger.info(day)
        for hour in [3,9,15,21]: #
            ymd=utils.to_datetime(day).strftime("%Y%m%d")
            url="http://west.rssoffice.com:8080/thredds/dodsC/roms/CA3km-nowcast/CA/ca_subCA_das_%s%02d.nc"%(ymd,hour)
            base_fn=url.split('/')[-1]
            local_fn=os.path.join(cache_path,base_fn)
            if os.path.exists(local_fn):
                logger.info("%30s: exists"%base_fn)
            else:
                logger.info("%30s: fetching"%base_fn)
                ds_fetch=xr.open_dataset(url)
                # in the future, we could subset at this point for a possible speedup.
                logger.info("%30s  saving"%"")
                ds_fetch.to_netcdf(local_fn)
                ds_fetch.close()
                logger.info("%30s  sleeping"%"")
                time.sleep(5) # be nice!
            local_files.append(local_fn)
    return local_files

   
# Choose a subset of that to make into the DFM domain
# picked off a map

def extract_roms_subgrid(ul_xy=(358815., 4327282.),
                         lr_xy=(633421., 4009624.)):
    """
    Loads CA ROMS 4km output
    selects a subset of that grid based on the UTM extents specified
    creates a curvilinear unstructured grid matching that subset of the ROMS
    grid
    projects to UTM, trims dry cells (based on ROMS data).
    cleans up the numbering and returns that grid, without bathymetry

    cells['lati'] and ['loni'] had been indexes to just the subset of the ROMS 
    grid.  Those are now indexes into the original roms grid.
    """

    snap_fns=glob.glob(os.path.join(cache_path,'*.nc'))
    
    snap_fns.sort()

    ds=xr.open_dataset(snap_fns[0])

    ds0=ds.isel(time=0)

    ul_ll=utm2ll(ul_xy)
    lr_ll=utm2ll(lr_xy)

    lon_range=[ul_ll[0], lr_ll[0]]
    lat_range=[lr_ll[1], ul_ll[1]]

    lat_sel=(ds0.lat.values>=lat_range[0]) & (ds0.lat.values<=lat_range[1])
    lon_sel=(ds0.lon.values>=(lon_range[0]%360)) & (ds0.lon.values<=(lon_range[1]%360))
    # starting indices 
    lat0i=np.nonzero(lat_sel)[0][0]
    lon0i=np.nonzero(lon_sel)[0][0]
    ds0_sub=ds0.isel( lat=lat_sel, lon=lon_sel )

    Lat,Lon=np.meshgrid( ds0_sub.lat.values, ds0_sub.lon.values)

    ll_mesh=np.array( [Lon,Lat] ).transpose(1,2,0)

    xy_mesh=ll2utm(ll_mesh)

    # Extract a grid:

    g=unstructured_grid.UnstructuredGrid()

    # pretty sure that the specified lon/lat are for cell centers (rho points)
    node_lat=utils.center_to_edge(ds0_sub.lat.values)
    node_lon=utils.center_to_edge(ds0_sub.lon.values)

    # Maybe 20s
    mappings = g.add_rectilinear(p0=[node_lon[0],node_lat[0]],
                                 p1=[node_lon[-1],node_lat[-1]],
                                 nx=len(node_lon),
                                 ny=len(node_lat))
                                
    # Remove dry cells and store indices to get back to lat/lon grid
    dry=np.isnan(ds0_sub.zeta.values)

    cell_values=np.zeros(g.Ncells())
    cell_lati=np.zeros(g.Ncells(),'i4')
    cell_loni=np.zeros(g.Ncells(),'i4')

    for lati,lat in enumerate(ds0_sub.lat.values):
        for loni,lon in enumerate(ds0_sub.lon.values):
            celli=mappings['cells'][loni,lati]
            # These are now referenced to the full ROMS grid
            cell_lati[celli]=lati + lat0i
            cell_loni[celli]=loni + lon0i
            if dry[lati,loni]:
                cell_values[ celli ] = 1.0

    g.add_cell_field('lati',cell_lati)
    g.add_cell_field('loni',cell_loni)

    # Reproject to UTM zone 10 meters
    g.nodes['x']=proj_utils.mapper('WGS84','EPSG:26910')(g.nodes['x'])

    # trim the dry cells:
    for c in np.nonzero(cell_values)[0]:
        g.delete_cell(c)

    # Clean up the nodes and edges related to deleted cells,
    # get everything consecutively numbered
    g.renumber_cells()
    g.make_edges_from_cells()
    g.delete_orphan_nodes()
    g.renumber()
    return g

def extract_roms_subgrid_poly(poly):
    """
    Extract a ROMS subgrid based on a polygon (in UTM coordinates)
    """
    # Start with an oversized, rectangular grid:
    g=extract_roms_subgrid( ul_xy=(poly.bounds[0],poly.bounds[3]),
                            lr_xy=(poly.bounds[2],poly.bounds[1]))
    # Then trim the fat:
    to_delete=~g.select_cells_intersecting(poly)
    for c in np.nonzero(to_delete)[0]:
        g.delete_cell(c)

    # And clean up
    g.renumber_cells()
    g.make_edges_from_cells()
    g.delete_orphan_nodes()
    g.renumber()
    return g

# 6k cells.  Not bad.
@memoize.memoize()
def coastal_dem():
    # Note - this is not the CA ROMS cache path, since it's not referencing
    # a CA ROMS dataset, but a more general bathy dataset
    fn=os.path.join(local_config.cache_path,"ngdc-etopo-merge-utm.tif")

    assert os.path.exists(fn)

    # The recipe for making that file
    # if not os.path.exists(fn):
    #     crm_ll=field.GdalGrid('ngdc-crm.tif')
    #     missing=(crm_ll.F==-32768)
    #     crm_ll.F=crm_ll.F.astype('f8')
    # 
    #     dem_etopo=field.GdalGrid('etopo1.tif')
    # 
    #     X,Y = crm_ll.XY()
    #     bad_x=X[missing]
    #     bad_y=Y[missing]
    # 
    #     fill=dem_etopo( np.c_[bad_x,bad_y] )
    #     crm_ll.F[missing] = fill
    # 
    #     dem=crm_ll.warp(t_srs='EPSG:26910',
    #                     s_srs='WGS84',
    #                     fn=fn)
    # else:
    #     dem=field.GdalGrid(fn)
    return field.GdalGrid(fn)
    

def add_coastal_bathy(g,dem=None):
    dem=dem or coastal_dem()
    
    # Add depth to the grid:
    def clip(x):
        return x.clip(-np.inf,-4)
    
    node_depth=clip( dem( g.nodes['x'] ) )
    cell_depth=clip( dem( g.cells_centroid() ) )
    edge_depth=clip( dem( g.edges_center() ) )

    # the redundant names are so it can get written to ugrid
    # where all variables have a single namespace, and when it
    # comes back in we will still have node depth called 'depth'
    g.add_node_field('depth',node_depth)
    g.add_edge_field('edge_depth',edge_depth)
    g.add_cell_field('cell_depth',cell_depth)

    return g

def annotate_grid_from_data(g,start,stop,candidate_edges=None):
    """
    Add src_idx_in,src_idx_out fields to edges for ROMS-adjacent boundary 
    edges.

    g: unstructured_grid to be annotated
    start,stop: datetime64 date range, for selecting wet cells from ROMS
    candidate_edges: if specified, only consider these edges, an array
      of edge indices
    """
    # Get the list of files
    ca_roms_files=fetch_ca_roms(start,stop)
    
    # Scan all of the ROMS files to find cells which are always wet
    wet=True
    for ca_roms_file in ca_roms_files:
        logging.info(ca_roms_file)
        ds=xr.open_dataset(ca_roms_file)
        wet = wet & np.isfinite( ds.zeta.isel(time=0).values )
        ds.close()

    # If we had ROMS data in a single dataset:    
    # wet=np.all( np.isfinite(src.zeta.values),
    #             axis=src.zeta.get_axis_num('time') )

    boundary_cells=[]
    boundary_edges=[]

    # record indices into src for the src cells just inside and
    # outside each boundary edge, or -1 for non-boundary
    edge_src_idx_in=np.zeros( (g.Nedges(),2),'i4') - 1
    edge_src_idx_out=np.zeros( (g.Nedges(),2),'i4') - 1
    edge_norm_in=np.zeros( (g.Nedges(),2),'f8')*np.nan

    N=g.Nedges()
    g.edge_to_cells() # Make sure that's been populated

    centroid=g.cells_centroid()
    edge_ctr=g.edges_center()

    src=xr.open_dataset(ca_roms_files[0])

    if candidate_edges is None:
        candidate_edges=np.arange(g.Nedges())
    
    for j in candidate_edges: # range(g.Nedges()):
        if j%1000==0:
            logger.info("%d/%d"%(j,N))
        c1,c2=g.edges['cells'][j]
        if (c1>=0) and (c2>=0):
            continue # not a boundary in this grid
        if c1<0:
            cin=c2
        else:
            cin=c1
        # A bit goofy, but make this a bit more general...
        # Construct the center of the cell that would be our neighbor:
        cout_cc=2*edge_ctr[j] - centroid[cin]
        cout_cc_ll=utm2ll(cout_cc)

        lon_idx_out=utils.nearest(src.lon.values,cout_cc_ll[0]%360.0)
        lat_idx_out=utils.nearest(src.lat.values,cout_cc_ll[1])

        if not wet[lat_idx_out,lon_idx_out]:
            continue

        # good - it's a real boundary edge
        cin_cc_ll=utm2ll(centroid[cin])
        lon_idx_in=utils.nearest(src.lon.values,cout_cc_ll[0]%360.0)
        lat_idx_in=utils.nearest(src.lat.values,cout_cc_ll[1])

        boundary_edges.append(j)
        boundary_cells.append(cin)
        edge_src_idx_out[j]=(lat_idx_out,lon_idx_out) # assumes an order in src
        edge_src_idx_in[j]=(lat_idx_in,lon_idx_in) # assumes an order in src

        edge_norm_in[j,:] = utils.to_unit( centroid[cin] - cout_cc )

    g.add_edge_field('src_idx_out',edge_src_idx_out,on_exists='overwrite')
    g.add_edge_field('src_idx_in',edge_src_idx_in,on_exists='overwrite')
    g.add_edge_field('bc_norm_in',edge_norm_in,on_exists='overwrite')
    

def set_ic_from_map_output(snap,map_file,output_fn='initial_conditions_map.nc',
                           missing=0,tol_km=10):
    """
    copy the structure of the map_file at one time step, but overwrite
    fields with ROMS snapshot data

    snap: A ROMS output file, loaded as xr.Dataset
    map_file: path to map output, must be single processor run.
    output_fn: If specified, the updated map file is written to the given path.
    
    missing: for locations in the map file which are missing or unmatched in the ROMS
    file, set scalars to this value.  If set to None, leave those water columns
    as is in the map file.

    tol_km: if the best ROMS match is farther away than this, set it to missing

    returns a Dataset() of the updated map
    """
    dest_fields=['sa1','tem1'] # names of the fields to write to in the map file
    roms_fields=['salt','temp'] # source fields in the ROMS data

    map_in=xr.open_dataset(map_file)

    map_out=map_in.isel(time=[0])
    # there are some name clashes -- drop any coordinates attributes
    #
    for dv in map_out.data_vars:
        if 'coordinates' in map_out[dv].attrs:
            del map_out[dv].attrs['coordinates']

    # DFM reorders the cells, so read it back in.
    g_map=dfm_grid.DFMGrid(map_out)
    g_map_cc=g_map.cells_centroid()
    g_map_ll=utm2ll(g_map_cc)

    ## 
    dlon=np.median(np.diff(snap.lon.values))
    dlat=np.median(np.diff(snap.lat.values))

    roms_z=-snap.depth.values[::-1]

    snap0=snap.isel(time=0)
    # make the dimension order explicit so we can safely index
    # this via numpy below
    wet=np.isfinite(snap0.zeta.transpose('lat','lon').values)
    snap_scalars=[snap0[roms_field].transpose('lat','lon','depth').values
                  for roms_field in roms_fields]
    #snap_salt=snap0.salt.transpose('lat','lon','depth').values
    
    snap_lon=snap0.lon.values
    snap_lat=snap0.lat.values

    sel_time=xr.DataArray([0],dims=['time'])
    sel_cell=xr.DataArray([1000000],dims=['nFlowElem'])

    all_bl=map_out.FlowElem_bl.values
    all_wd=map_out.waterdepth.isel(time=0).values
    # This only works for uniform sigma values - spatially variable or
    # z-levels would require other code
    sigma=map_out.LayCoord_cc.values
    assert sigma.ndim==1
    
    def get_dfm_z(c):
        # This way is nice but very slow:
        # bl=map_out.FlowElem_bl.isel(nFlowElem=c).values # positive up from the z datum
        # wd=map_out.waterdepth.isel(nFlowElem=c,time=0).values
        # sigma=map_out.LayCoord_cc.values
        bl=all_bl[c]
        wd=all_wd[c]

        return bl + wd*sigma
    
    # much faster to write straight to numpy array rather than
    # via xarray.  But for a bit more robustness, hack together
    # dynamic indexing

    dest_arrays=[map_out[fld].values
                 for fld in dest_fields]
    # dest_salt=map_out.sa1.values
    dest_idx=[]
    for dimi,dim in enumerate(map_out.sa1.dims):
        if dim=='time':
            dest_idx.append(0)
        elif dim=='laydim':
            dest_idx.append(slice(None))
        else:
            dest_idx_cell=dimi
            dest_idx.append(100000)
            
    for c in g_map.valid_cell_iter():
        if c%1000==0:
            print("%d/%d"%(c,g_map.Ncells()))
        is_missing=False
        
        loni=utils.nearest(snap_lon%360,g_map_ll[c,0]%360)
        lati=utils.nearest(snap_lat,g_map_ll[c,1])
        
        err_km=utils.haversine([snap_lon[loni],snap_lat[lati]],
                               g_map_ll[c,:])
        
        if err_km>tol_km:
            is_missing=True
        elif not bool(wet[lati,loni]):
            is_missing=True
        else:
            # grab a water column, and flip it vertically to be
            # in order of increasing, positive-up, depth, i.e.
            # bed to surface.
            # xarray = slow
            #   roms_salt=snap0.salt.isel(lat=lati,lon=loni).values[::-1]
            # numpy = fast
            roms_scalars=[ snap_scalar[lati,loni,::-1]
                           for snap_scalar in snap_scalars]
            # roms_salt=snap_salt[lati,loni,::-1]

            #valid=np.isfinite(roms_salt)
            # Assume that if the first (salt) is valid, then so are the rest (temperature)
            valid=np.isfinite(roms_scalars[0])
            if not np.any(valid):
                is_missing=True 
            else:
                dfm_z=get_dfm_z(c)

                dfm_scalars=[ np.interp(dfm_z,roms_z[valid],roms_scalar[valid])
                              for roms_scalar in roms_scalars ]
                #dfm_salt=np.interp(dfm_z,
                #                   roms_z[valid],roms_salt[valid])

        dest_idx[dest_idx_cell]=c
        if is_missing:
            if missing is None:
                continue
            else:
                for dest in dest_arrays:
                    dest[dest_idx]=missing
                # dest_salt[dest_idx]=missing
        else:
            #dest_salt[dest_idx]=dfm_salt
            for dest,dfm_scalar in zip(dest_arrays,dfm_scalars):
                dest[dest_idx]=dfm_scalar

    #map_out.sa1.values[:,:,:]=dest_salt # a little dicey
    for dest_field,dest in zip(dest_fields,dest_arrays):
        map_out[dest_field].values[:,:,:]=dest # a little dicey        
    # that was legal, and took, right? check the first one
    assert np.allclose( map_out.sa1.values, dest_arrays[0] )
    
    # Does the map timestamp have to match what's in the mdu?
    # Yes.
    if output_fn is not None:
        map_out.to_netcdf(output_fn,format='NETCDF3_64BIT')
    return map_out

# Not exactly ROMS-specific, but helps with using ROMS coupling
def add_sponge_layer(mdu,run_base_dir,grid,edges,sponge_visc,background_visc,sponge_L,
                     quantity='viscosity'):
    """
    quantity: 'viscosity' or 'diffusivity'
    """
    obc_centers=grid.edges_center()[edges]

    # sponge length scale: 25000 [m] is roughly 8 cells

    sample_sets=[ np.c_[obc_centers[:,0],obc_centers[:,1],sponge_visc*np.ones(len(obc_centers))] ]

    circs=[geometry.Point(xy).buffer(sponge_L)
           for xy in obc_centers]
    obc_buff=cascaded_union(circs).boundary
    obc_buff_pnts=np.array(obc_buff)
    obc_buff_pnts_resamp=linestring_utils.downsample_linearring(obc_buff_pnts,sponge_L*0.5)

    sample_sets.append( np.c_[obc_buff_pnts_resamp[:,0],
                              obc_buff_pnts_resamp[:,1],
                              background_visc*np.ones(len(obc_buff_pnts_resamp))] )

    # And some far flung values
    x0,x1,y0,y1=grid.bounds()
    corners=np.array( [[x0-sponge_L,y0-sponge_L],
                       [x0-sponge_L,y1+sponge_L],
                       [x1+sponge_L,y1+sponge_L],
                       [x1+sponge_L,y0-sponge_L]] )

    sample_sets.append( np.c_[corners[:,0],
                              corners[:,1],
                              background_visc*np.ones(len(corners))] )

    visc_samples=np.concatenate(sample_sets,axis=0)

    np.savetxt(os.path.join(run_base_dir,'%s.xyz'%quantity),
               visc_samples)

    txt="\n".join(["QUANTITY=horizontaleddy%scoefficient"%quantity,
                   "FILENAME=%s.xyz"%quantity,
                   "FILETYPE=7",
                   "METHOD=4",
                   "OPERAND=O"
                   "\n"])
    
    old_bc_fn = os.path.join(run_base_dir,mdu['external forcing','ExtForceFile'])
    with open(old_bc_fn,'at') as fp:
        fp.write(txt)
        
