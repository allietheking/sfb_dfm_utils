import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from stompy.spatial import interp_4d, proj_utils

from stompy.grid import unstructured_grid
from stompy.model.delft import dfm_grid
from stompy.spatial import wkb2shp
from stompy import utils
from stompy.model import unstructured_diffuser

from stompy.io.local import usgs_sfbay

from . import common # cache_dir

def add_initial_salinity(run_base_dir,
                         static_dir,
                         old_bc_fn,
                         all_flows_unit=False):
    static_dir_rel=os.path.relpath(static_dir,run_base_dir)
    
    # Spatial salinity initial condition and friction
    lines=[]
    if not all_flows_unit: # real initial condition:
        lines+=[ "QUANTITY=initialsalinity",
                 "FILENAME=%s/orig-saltopini.xyz"%static_dir_rel,
                 "FILETYPE=7",
                 "METHOD=5",
                 "OPERAND=O",
                 ""]
    else: #  constant 35 ppt initial condition:
        print("-=-=-=- USING 35 PPT WHILE TESTING! -=-=-=-")
        lines+=[ "QUANTITY=initialsalinity",
                 "FILENAME=constant_35ppt.xyz",
                 "FILETYPE=7",
                 "METHOD=5",
                 "OPERAND=O",
                 ""]
        orig_salt=np.loadtxt(os.path.join(static_dir,'orig-saltopini.xyz'))

        orig_salt[:,2]=35
        np.savetxt(os.path.join(run_base_dir,'constant_35ppt.xyz'),
                   orig_salt,
                   delimiter=' ')
    with open(old_bc_fn,'at') as fp:
        fp.write("\n".join(lines))


# fancier version pulls some data and extrapolates


##

def samples_to_cells(init_salt,g):
    """
    init_salt: array [N,3] samples point * {x,y,salt}  
    g: the unstructured_grid to extrapolate onto
    """
    # try again, but with the interp_4d code:
    samples=pd.DataFrame()
    samples['x']=init_salt[:,0]
    samples['y']=init_salt[:,1]
    samples['value']=init_salt[:,2]
    # doesn't really matter, though should be kept in check with alpha
    # and K_j
    samples['weight']=1e6*np.ones_like(init_salt[:,0])

    # alpha=2e-5 is too sharp
    # 5e-6 still too sharp
    # Not sure why higher values looked fine when running this directly,
    # but when it's part of the sfb_dfm.py script it needs really low
    # values of alpha.
    salt=interp_4d.weighted_grid_extrapolation(g,samples,alpha=3e-7)

    # If these fail, the extrapolation approach may be running into
    # numerical difficulties, often made worse by an alpha which is too
    # large (which might be attempting to do less smoothing).
    # decreasing alpha (which results in a smoother field) may help.
    assert np.all( np.isfinite(salt) )
    assert salt.max() < 40
    assert salt.min() > -1 # allow a bit of slop

    # use centroids as they are more predictable
    cc=g.cells_centroid()

    cc_salt=np.concatenate( ( cc, salt[:,None] ),axis=1 )
    return cc_salt


def samples_from_usgs_erddap(run_start,field='salinity'):
    """
    DEPRECATED.  Better to use the dynamically cached data from
    samples_from_usgs().

    --
    Pull some USGS transect data and return a set of 2D salinity
    samples appropriate for the given date.
    Caveats: This is currently using ERDDAP behind the scenes, which
    does not have the most recent data.  It will use a prior year
    if the date cannot be matched within 30 days.

    field: name of the field to pull from usgs_crusies.
       'salinity','temperature'
    """
    # This copy of the USGS data ends early:
    usgs_data_end=np.datetime64('2016-04-28')
    usgs_pad=np.timedelta64(30,'D')

    usgs_target=run_start

    # so we may have to grab a previous years cruise and pretend
    while usgs_target + usgs_pad > usgs_data_end:
        usgs_target -= np.timedelta64(365,'D')

    usgs_cruises=usgs_sfbay.cruise_dataset(usgs_target - usgs_pad,
                                           usgs_target + usgs_pad )

    # lame filling
    scal3d=usgs_cruises[field]
    scal2d=scal3d.mean(dim='prof_sample')
    assert scal2d.dims[0]=='date'
    scal2d_fill=utils.fill_invalid(scal2d.values,axis=0)

    scal_f=interp1d(utils.to_dnum(scal2d.date.values),
                    scal2d_fill,
                    axis=0,bounds_error=False)(utils.to_dnum(usgs_target))

    usgs_init_scal=np.c_[scal2d.x.values,scal2d.y.values,scal_f]
    return usgs_init_scal


def samples_from_usgs(run_start,field='salinity'):
    run_start=utils.to_dt64(run_start)
    pad=np.timedelta64(30,'D')
    
    df=usgs_sfbay.query_usgs_sfbay(period_start=run_start-pad,
                                   period_end=run_start+pad,
                                   cache_dir=common.cache_dir)

    # narrow to the columns we care about:
    if field=='salinity':
        dfield='Salinity'
    elif field=='temperature':
        dfield='Temperature'
    else:
        assert False,"Unknown field %s"%field
        
    df2=df.loc[:,['Date','Station Number','Depth',dfield]]
    # will have to get coordinates from elsewhere

    # depth average:
    df3=df2.groupby( ['Date','Station Number'] )[dfield].mean()

    run_start_dnum=utils.to_dnum(run_start)
    def time_interp(grp):
        dnums=[ utils.to_dnum(v[0]) for v in grp.index.values ]
        return np.interp( run_start_dnum, dnums, grp.values )

    ser4=df3.groupby('Station Number').apply(time_interp)

    lls=[ usgs_sfbay.station_number_to_lonlat(s)
          for s in ser4.index.values ]
    lls=np.array(lls)
    xys=proj_utils.mapper('WGS84','EPSG:26910')(lls)

    # glue those together to get [N,3] array, {x,y,salt}
    return np.c_[ xys, ser4.values ]


def samples_from_sfei_moorings(run_start,static_dir):
    """
    return [N,3] array of salinity data from SFEI moorings appropriate for
    given date.  Note that this may have no data, but will be returned
    as a [0,3] array
    """
    # And pull some SFEI data:

    mooring_xy=[]
    mooring_salt=[]

    L2_dir='/opt/data/sfei/moored_sensors_csv/L2/'

    # tuples (<name in observation points shapefile>, <L2 data file name> )
    sfei_moorings=[
        ('ALV',"ALV_all_data_L2.csv"),
        ('SFEI_Coyote',"COY_all_data_L2.csv"),
        ('DB',"DMB_all_data_L2.csv"),
        ('SFEI_Guadalupe',"GL_all_data_L2.csv"),
        ('SFEI_Mowry',"MOW_all_data_L2.csv"),
        ('SFEI_Newark',"NW_all_data_L2.csv"),
        ('SFEI_A8Notch',"POND_all_data_L2.csv"),
        ('SMB',"SM_all_data_L2.csv")
    ]

    # lat/lon from observation-points
    # FRAGILE - FIX!
    obs_shp=wkb2shp.shp2geom(os.path.join(static_dir,"observation-points.shp"))

    for name,l2_file in sfei_moorings:
        print(name)
        sfei=pd.read_csv(os.path.join(L2_dir,l2_file),
                         parse_dates=['Datetime','dt'],low_memory=False)
        sfei_salt=sfei['S_PSU']
        valid=~(sfei_salt.isnull())
        # limit to data within 20 days of the request
        sfei_salt_now=utils.interp_near(utils.to_dnum(run_start),
                                        utils.to_dnum(sfei.Datetime[valid]),sfei_salt[valid],
                                        max_dx=20.0)
        geom=obs_shp['geom'][ np.nonzero(obs_shp['name']==name)[0][0] ]
        xy=np.array(geom)
        if np.isfinite(sfei_salt_now):
            mooring_xy.append(xy)
            mooring_salt.append(sfei_salt_now)

    if len(mooring_xy):
        xy=np.array(mooring_xy)
        sfei_init_salt=np.c_[xy[:,0],xy[:,1],mooring_salt]
    else:
        sfei_init_salt=np.zeros( (0,3),'f8')
    return sfei_init_salt

def initial_salinity_dyn(run_base_dir,
                         mdu,
                         static_dir,
                         run_start):
    # Get some observations:
    usgs_init_salt=samples_from_usgs(run_start)

    mooring_salt=samples_from_sfei_moorings(run_start,static_dir=static_dir)

    init_salt=np.concatenate( (usgs_init_salt,
                               mooring_salt) )
    ##
    g=dfm_grid.DFMGrid(os.path.join(run_base_dir,mdu['geometry','NetFile']))

    # Above here is assembling init_salt
    # Below is extrapolating -- needs g
    cc_salt = samples_to_cells(init_salt,g)


    # Because DFM is going to use some interpolation, and will not reach outside
    # the convex hull, we have to be extra cautious and throw some points out farther
    # afield.

    xys_orig=np.loadtxt(os.path.join(static_dir,'orig-saltopini.xyz'))

    combined_xys=np.concatenate( (cc_salt,xys_orig), axis=0 )

    ##
    return combined_xys


def add_initial_salinity_dyn(run_base_dir,
                             static_dir,
                             mdu,run_start):
    
    # Spatial salinity initial condition 
    lines=[]
    lines+=[ "QUANTITY=initialsalinity",
             "FILENAME=saltopini.xyz",
             "FILETYPE=7",
             "METHOD=5",
             "OPERAND=O",
             ""]
    xys=initial_salinity_dyn(run_base_dir,
                             mdu,static_dir,run_start)
    np.savetxt(os.path.join(run_base_dir,'saltopini.xyz'),
               xys,
               delimiter=' ')
    old_bc_fn=os.path.join(run_base_dir,
                           mdu['external forcing','ExtForceFile'] )
        
    with open(old_bc_fn,'at') as fp:
        fp.write("\n".join(lines))



##

# Pull and cache USGS cruise data directly

