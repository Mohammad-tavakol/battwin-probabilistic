"""Slurry-basis component accounting; no renormalisation of user compositions."""
import math,re

COMPONENTS=('Am/total weight%','Conductive additive/total weight %','Binder1/total weight %','Binder2 (thickener)/total weight %')
SOLIDS='Solid Content (%)'
IDENTITIES={'Conductive additive/total weight %':'Conductive additive type','Binder1/total weight %':'Binder1 type','Binder2 (thickener)/total weight %':'Binder2 (Thickener) type'}

def key(name):return re.sub(r'[^a-z0-9]','',str(name).casefold())

def component_fields(values):
    lookup={key(c):c for c in values.keys()}
    return {c:lookup[key(c)] for c in COMPONENTS if key(c) in lookup}

def calculated_solids(values):
    fields=component_fields(values)
    missing=[c for c in COMPONENTS if c not in fields]
    if missing:raise ValueError('Provide all formulation fractions to calculate solid content: '+', '.join(missing))
    amounts=[float(values[fields[c]]) for c in COMPONENTS]
    if any(not math.isfinite(v) or v<0 or v>100 for v in amounts):raise ValueError('Formulation fractions must be finite percentages between 0 and 100.')
    total=round(math.fsum(amounts),12)
    if not 0<total<=100:raise ValueError('The sum of active material, conductive additive and binder fractions must be greater than 0 and at most 100%.')
    return total

def with_calculated_solids(values):
    result=dict(values)
    total=calculated_solids(result)
    name=next((c for c in result if key(c)==key(SOLIDS)),SOLIDS)
    result[name]=total
    return result

def validate_identities(values):
    lookup={key(c):v for c,v in values.items()}
    for component,identity in IDENTITIES.items():
        if key(identity) not in lookup or key(component) not in lookup:continue
        material=str(lookup[key(identity)]).strip(' \"\'').casefold()
        if material in ['none','not used','nan',''] and float(lookup[key(component)])>0:
            raise ValueError(component+' is positive but its material is Not used. Select a material or set its fraction to zero.')

def validate_formulation(values):
    calculated_solids(values);validate_identities(values)

def prepare_formulation(frame,require_complete=True):
    """Copy inputs and derive solids from four slurry-basis fractions.

    Partial native feature tables for individual models may retain their
    supplied solid content with require_complete=False. No fractions are guessed.
    Complete formulations always override a stale supplied solid-content value.
    """
    import pandas as pd
    if not isinstance(frame,pd.DataFrame):raise TypeError('Provide a pandas DataFrame')
    fields=component_fields(frame)
    if len(fields)!=len(COMPONENTS):
        if require_complete:
            missing=[c for c in COMPONENTS if c not in fields]
            raise ValueError('Provide all formulation fractions to calculate solid content: '+', '.join(missing))
        return frame.copy()
    result=frame.copy();totals=[]
    for index,row in frame.iterrows():
        try:
            total=calculated_solids(row);validate_identities(row)
        except ValueError as exc:raise ValueError(f'Invalid formulation at row {index}: {exc}') from exc
        totals.append(total)
    names=[c for c in frame if key(c)==key(SOLIDS)] or [SOLIDS]
    for name in names:result[name]=totals
    return result
