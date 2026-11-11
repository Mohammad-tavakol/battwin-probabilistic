"""Target-specific feature recipes for density-aware process ensembles."""
import numpy as np
import pandas as pd

MASS='Mass loading dry (mg/cm2)'
DRY_T='Dry Coating Thickness (\u00b5m)'
DRY_P='Dried Porosity (%)'
RHO='Solid density (g/cm3)'

def density_inputs(frame):
    result=frame.copy()
    if RHO not in result:
        if 'Chemistry' in result:
            value=result.Chemistry.astype(str).str.strip().str.strip("'\"").str.upper().map({'NMC622':4.447,'NMC111':4.770,'GRA':2.158})
        else:value=pd.Series(np.nan,index=result.index)
        if 'Solid density reference (g/cm3)' in result:value=value.fillna(result['Solid density reference (g/cm3)'])
        if 'Electrode type' in result:
            fallback=result['Electrode type'].astype(str).str.strip().str.strip("'\"").str.casefold().map({'cathode':4.447,'anode':2.158})
            value=value.fillna(fallback)
        result[RHO]=value
    result[RHO]=pd.to_numeric(result[RHO],errors='raise').astype(float)
    if not np.isfinite(result[RHO]).all() or not result[RHO].gt(0).all():
        raise ValueError('Supply a positive solid density in g/cm3 for every material.')
    return result

def engineered_inputs(frame,stage):
    result=density_inputs(frame)
    if stage=='coating_drying':
        result['nominal_wet_film_um']=result['Coating Gap (micron)']/(result['Coating ratio (%)']/100)
        result['wet_solids_mass_proxy']=.1*result['slurry density g/cm3']*result['nominal_wet_film_um']*result['Solid Content (%)']/100
        result['drying_temperature_difference']=result['Drying temp-Zone2']-result['Drying temp-Zone1']
        result['inverse_coating_speed']=1/result['Coating Speed (m/min)']
        if MASS in result:result['predicted_dry_mass_loading']=result[MASS]
        if DRY_T in result:result['predicted_thickness']=result[DRY_T]
    elif stage=='calendaring':
        # Always use the supplied upstream response, never silently replace it
        # with the mass/thickness equation. The two features are complements.
        result['dry_porosity_consistent']=result[DRY_P]
        result['dry_bulk_solid_fraction']=1-result[DRY_P]/100
        result['gap_to_dry_thickness']=result['Calendaring Gap (um)']/result[DRY_T]
        result['nominal_gap_porosity']=100*(1-10*result[MASS]/(result[RHO]*result['Calendaring Gap (um)']))
    return result
