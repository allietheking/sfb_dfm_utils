# -*- coding: utf-8 -*-
"""
plotting utility for DFM run. trying to take advantage of rusty's stompy utilities. 

if you point this to the MDU file it should do the rest!!! (fingers crossed)

this script works by reading in the PLI and TIM files. The .pli file has all the geometric/ geographic 
information about the boundary condition while the tim file has the time series with data. 

@author: siennaw
"""

import sys
import matplotlib
matplotlib.use('agg',warn=False, force=True)    # need to do this so we can re-load pyplot with new backend
sys.path.append(r'/hpcvol1/siennaw/lib/stompy//')   # Linking to my stompy for now
import stompy.model.delft.io as dio
from pathlib import Path
matplotlib.pyplot.switch_backend('agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# MAIN INPUT: the mdu filename     
# mdu_filename = r'/hpcvol2/Open_Bay/Hydro_model/Full_res/WY2017/wy2017_usgs_bc/wy2017_usgs_bc.mdu' #<<<< input here

def plot_MDU(mdu_filename, gridpath): 
    #------------- script now takes over -------------------------------------------
    mdu_filename = Path(mdu_filename)
    base_dir     = mdu_filename.parent  # The assumption is that we'll find all our bc's in the same folder as the mdu.
    folder_dir   = base_dir / 'bc_figures'
    folder_dir.exists() or folder_dir.mkdir() 
    
    # Load in the grid  (assumption that it is the same grid.)
    from stompy.grid import unstructured_grid
    grid    = str(gridpath) # Load in shapefile of SFB 
    grid    = unstructured_grid.UnstructuredGrid.read_dfm(grid, cleanup=True)
    
    # Open MDU, strip time information using stompy functionality
    MDU = dio.MDUFile(filename=str(mdu_filename))
    t_ref, t_start, t_stop =  MDU.time_range() 
    
    
    # define shared plotting functions 
    def format_xaxis (axis):
        months = mdates.MonthLocator(interval = 2)  # every other month
        fmt = mdates.DateFormatter('%b/%Y')
        axis.xaxis.set_major_locator(months)
        axis.xaxis.set_major_formatter(fmt)
        axis.set_xlim(t_ref, t_stop)
        
    def save_image(fig, name):
        fullname = folder_dir / (name + '.png')
        fig.savefig(str(fullname), dpi = 300, bbox_inches='tight')
        print('Saved %s' % fullname)
        plt.close()
    
    # Section one
    #  Let's first read through the source_files (which seem to be the POTWs)
    
    sourcefolder = base_dir / 'source_files'
    PLIs = list(sourcefolder.glob('*.pli'))  # get a list of all the pli files in the directory
    
    # Iterate through each one. Note each pli file has a corresponding timeseries of data (*.tim)
    for bc in PLIs:
        print('Reading %s' % bc.stem)
        pli = dio.read_pli(str(bc))                      # read in the *.pli file  
        tim_filename = sourcefolder / (bc.stem + '.tim') # filename of corresponding timeseries
        tim = dio.read_dfm_tim(str(tim_filename), t_ref, time_unit='M', columns = ['flow','sal','temp']) 
    
        # Plot the data 
        fig = plt.figure(figsize=(11, 3))
        ax1 = fig.add_axes([0.05, 0.05, 0.68, 0.8])
        map_axis = fig.add_axes([0.55, 0.18, 0.6, 0.6])
        name = pli[0][0]
        ax1.set_title( name.capitalize() + ' (POTW Source)')
        ax1.plot(tim.time, tim.flow,'-', linewidth = 5, alpha = 0.5, color = 'skyblue')    
        ax1.grid(b = True, alpha = 0.25)
        ax1.set_ylabel("Flow (m$^3$/s)")
        format_xaxis(ax1)
        
        # Plot SFB map + location  
        grid.plot_edges(ax = map_axis, alpha = 0.8) 
        map_axis.axis('off')
        coords = pli[0][1] 
        for coord in coords:
            x, y = coord[0], coord[1] # There is a z coordinate we are ignoring here 
            map_axis.plot(x , y,'o', markersize= 11, color= 'orangered')
    
        # Quick check that temp/salinity are fixed:
        temp = set(tim.temp.values)
        sal  = set(tim.sal.values)
        if len(temp)>1 or len(sal)>1:
            print('sal or temp is NOT FIXED at %s' % bc.stem)
        else:
            label = 'Temperature is fixed at %d C\n Salinity is fixed at %d ppt' %  (temp.pop(), sal.pop())
            ax1.text(1.08, .05, label,  horizontalalignment='left',  verticalalignment='center', transform=ax1.transAxes, fontsize = 12)
        save_image(fig, name)
        
        
    '''
    NEXT : ONTO THE BOUNDARY CONDITIONS FOR THE INFLOWS / STREAMS CREEKS ETC
    
    The only tricky difference here is that these bc's are sometimes divided across multiple cells (aka,
    a big river might be split across 2 cell segments.... in this case, we look at the pli file (geometry) to see
    how many cells the BC is split across and then multiple discharge by the #/cells. We don't need to touch
    temperature (scalar) or salinity (concentration). 
    
    DFM should always divide evenly across cells (1/3 for 3 cells, 1/2 for 2 cells, so unless someone's 
    really decided to get creative with custom settings this assumption should hold)
    '''
    bcfolder = base_dir / 'bc_files'
    PLIs = list(bcfolder.glob('*.pli'))           
    for bc in PLIs:
    
        print('Reading %s' % bc.stem)
        # the way this works is that the bc is divided between multiple cells evenly. so we just take one and multiply by the number of poitns. 
        pli = dio.read_pli(str(bc))
        filenames  = pli[0][2]
        ncells = len(filenames)
        tim_filename = bcfolder / (filenames[0] + '.tim') # filename of corresponding timeseries
        tim = dio.read_dfm_tim(str(tim_filename), t_ref, time_unit='M', columns  = ['data']) #, columns = ['flow','sal','temp']) 
    
        # Plot the data 
        fig = plt.figure(figsize=(11, 3))
        ax1 = fig.add_axes([0.05, 0.05, 0.68, 0.8])
        map_axis = fig.add_axes([0.55, 0.18, 0.6, 0.6])
        name = pli[0][0]            # Name of the boundary condition
        ax1.set_title( name.capitalize() + ' (non-POTW source)')
        
        if 'flow' in name:
            ax1.set_ylabel("Flow (m$^3$/s)")
            tim.data.values = tim.data.values * ncells # multiply by # of segements inflow is divided across 
        elif 'salt' in name:
            ax1.set_ylabel("Salinity (PPT)")
        elif 'temp' in name:
            ax1.set_ylabel("Temperature (deg C)")
        elif 'ssh' in name:
            ax1.set_ylabel('Sea Surface Height Forcing (m)')
            
        format_xaxis(ax1)
        ax1.plot(tim.time, tim.data,'-', linewidth = 5, alpha = 0.5, color = 'skyblue')    
        ax1.grid(b = True, alpha = 0.25)
        # Plot SFB map + location  
        grid.plot_edges(ax = map_axis, alpha = 0.8) 
        map_axis.axis('off')
        coords = pli[0][1] 
        for coord in coords:
            x, y = coord[0], coord[1] # There is a z coordinate we are ignoring here
            map_axis.plot(x, y, 'o', markersize= 11, color= 'orangered')
        save_image(fig, name)
            
        print('Done plotting boundary conditions.')
    




  