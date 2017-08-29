import os
import numpy as np

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
