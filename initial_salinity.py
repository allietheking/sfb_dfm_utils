import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from stompy.grid import unstructured_grid
from stompy.model.delft import dfm_grid
from stompy.spatial import wkb2shp
from stompy import utils
from stompy.model import unstructured_diffuser

from stompy.io.local import usgs_sfbay

def add_initial_salinity(run_base_dir,
                         static_dir,
                         old_bc_fn,
                         all_flows_unit=False):
    static_dir_rel=os.path.relpath(static_dir,run_base_dir)
    
    # Spatial salinity initial condition and friction
    lines=[]
    if not all_flows_unit: # real initial condition:
        lines+=[ "QUANTITY=initialsalinity",
                 "FILENAME=%s/saltopini.xyz"%static_dir_rel,
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
        orig_salt=np.loadtxt(os.path.join(static_dir,'saltopini.xyz'))
        orig_salt[:,2]=35
        np.savetxt(os.path.join(run_base_dir,'constant_35ppt.xyz'),
                   orig_salt,
                   delimiter=' ')
    with open(old_bc_fn,'at') as fp:
        fp.write("\n".join(lines))


# fancier version pulls some data and extrapolates


##

# # Note - this code is fragile and assumes it is being run in lsb_dfm, not
# # in sfb_dfm_utils.
# run_start=np.datetime64('2015-12-15')
# saltopini_xyz=np.loadtxt('inputs-static/saltopini.xyz')
# mdu={('geometry','NetFile'):'lsb_v99_net.nc'}
# run_base_dir="runs/short_winter2016_04"

def initial_salinity_dyn(run_base_dir,
                         mdu,
                         static_dir,
                         run_start):
    g=dfm_grid.DFMGrid(os.path.join(run_base_dir,mdu['geometry','NetFile']))

    ##

    # Get some observations:

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
    salt3d=usgs_cruises['salinity']
    salt2d=salt3d.mean(dim='prof_sample')
    assert salt2d.dims[0]=='date'
    salt2d_fill=utils.fill_invalid(salt2d.values,axis=0)

    salt_f=interp1d(utils.to_dnum(salt2d.date.values),
                    salt2d_fill,
                    axis=0,bounds_error=False)(utils.to_dnum(usgs_target))

    usgs_init_salt=np.c_[salt2d.x.values,salt2d.y.values,salt_f]
    ##

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
                         parse_dates=['Datetime'])
        sfei_salt=sfei['S_PSU']
        valid=~(sfei_salt.isnull())
        sfei_salt_now=np.interp(utils.to_dnum(run_start),
                                utils.to_dnum(sfei.Datetime[valid]),sfei_salt[valid])
        geom=obs_shp['geom'][ np.nonzero(obs_shp['name']==name)[0][0] ]
        xy=np.array(geom)
        if np.isfinite(sfei_salt_now):
            mooring_xy.append(xy)
            mooring_salt.append(sfei_salt_now)

    ##

    xy=np.array(mooring_xy)

    sfei_init_salt=np.c_[xy[:,0],xy[:,1],mooring_salt]

    ##

    init_salt=np.concatenate( (usgs_init_salt,
                               sfei_init_salt) )

    ##


    differ=unstructured_diffuser.Diffuser(grid=g)

    ##

    for x,y,salt in init_salt:
        try:
            differ.set_dirichlet(salt,xy=[x,y],on_duplicate='skip')
        except differ.PointOutsideDomain as exc:
            print(exc)
            continue

    ##
    differ.construct_linear_system()
    differ.solve_linear_system(animate=False)

    ##

    cc=g.cells_centroid()

    ##

    cc_salt=np.concatenate( ( cc, differ.C_solved[:,None] ),axis=1 )

    ##

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
