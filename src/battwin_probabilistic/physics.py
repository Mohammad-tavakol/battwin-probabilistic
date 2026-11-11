"""Stage physical-consistency and sanity rules."""
import numpy as np
import pandas as pd
FINAL_TARGETS=["Gravimetric capacity (mA h/g)","Volumetric capacity (mA h/cm3)"]
def physical_stage_flags(particles,stages):
    flags={s.name:np.zeros(len(particles),dtype=bool) for s in stages};records=[]
    def rule(stage,name,passed):
        passed=np.asarray(passed,dtype=bool);flags[stage]|=~passed
        records.append(dict(Stage=stage,rule=name,n_particles=len(passed),n_failure=int((~passed).sum()),failure_probability=float((~passed).mean())))
    def val(name):return pd.to_numeric(particles[name],errors='coerce').to_numpy(float)
    for stage in stages:
        for output in stage.output_columns:
            v=val(output);rule(stage.name,output+' finite and nonnegative',np.isfinite(v)&(v>=0))
    dry='Dry Coating Thickness (µm)';mass='Mass loading dry (mg/cm2)';dryp='Dried Porosity (%)'
    thick='Calendered thickness (um)';por='Calendered porosity (%)'
    rule('Drying','Dried thickness < coating gap',val(dry)<val('Coating Gap (micron)'))
    rule('Drying','Positive dried thickness and density',(val(dry)>0)&(val('Solid density reference (g/cm3)')>0))
    rule('Drying','Dried porosity in [0,100]',(val(dryp)>=0)&(val(dryp)<=100))
    # These three threshold rules are the reference notebook's configured sanity constraints.
    rule('Drying','Reference minimum dried porosity > 30%',val(dryp)>30.)
    rule('Calendaring','Roll gap < dried thickness',val('Calendaring Gap (um)')<val(dry))
    rule('Calendaring','Calendered thickness < dried thickness',val(thick)<val(dry))
    rule('Calendaring','Positive calendered thickness',val(thick)>0)
    rule('Calendaring','Dried porosity > calendered porosity',val(dryp)>val(por))
    rule('Calendaring','Calendered porosity in [0,100]',(val(por)>=0)&(val(por)<=100))
    rule('Calendaring','Reference minimum calendered porosity > 16%',val(por)>16.)
    rule('Electrochemical','Reference gravimetric capacity < 500 mAh/g',val(FINAL_TARGETS[0])<500.)
    return flags,pd.DataFrame(records)


def porosity_from_thickness(mass_loading,thickness,solid_density,groups=None):
    """Porosity %, with optional within-condition mean measured mass for standalone checks.

    For the chain, pass corresponding predicted particle mass/thickness without grouping.
    This is a physics comparison, not a replacement for model fitting or a new calibration.
    """
    from .orchestration.core import calculate_physical_porosity
    mass,tau,rho=np.broadcast_arrays(np.asarray(mass_loading,float),np.asarray(thickness,float),np.asarray(solid_density,float))
    frame=pd.DataFrame(dict(mass=mass.ravel(),tau=tau.ravel(),rho=rho.ravel()))
    if groups is not None:frame['mass']=frame.groupby(np.asarray(groups),dropna=False).mass.transform('mean')
    return calculate_physical_porosity(frame,'mass','tau','rho',scale=10.,clip_bounds=None).reshape(mass.shape)
