"""Optional additive-volume estimate; input component percentages are slurry-basis.

No model or inference function calls this implicitly. A caller must explicitly
select estimated mode. Component defaults and units are documented in docs/SLURRY_DENSITY.md.
"""
import math
import re

FORMULA = 'mass_fraction_additive_volume_v1'
MODE = 'slurry_density_mode'
MATERIALS = 'slurry_density_materials'
MANUAL = 'manual_slurry_density_g_cm3'
DEFAULT_DENSITIES = dict(NMC622=4.447, NMC111=4.77, Graphite=2.26,
    C65=1.6, C45=1.6, PVDF=1.78, SBR=1.3, CMC=1.6, Water=1., NMP=1.03)
FALLBACK = 'Unknown material'
DEFAULT_DENSITIES[FALLBACK] = 1.6
NOTES = {
    'C45': 'Assumed 1.6; grade-specific value not established.',
    'PVDF': 'Representative polymer density; verify the actual grade.',
    'SBR': 'Dry polymer; dispersion water belongs in the solvent fraction.',
    'Water': 'Approximate room-temperature value.',
    'NMP': 'Approximate room-temperature value; verify grade and temperature.',
}
SLOTS = (
    ('Active material', 'Am/total weight%', 'Chemistry'),
    ('Conductive additive', 'Conductive additive/total weight %', 'Conductive additive type'),
    ('Binder 1', 'Binder1/total weight %', 'Binder1 type'),
    ('Binder 2 / thickener', 'Binder2 (thickener)/total weight %', 'Binder2 (Thickener) type'),
    ('Dispersant', 'Dispersant additive (%)', 'Dispersent Type'),
)
def key(value): return re.sub('[^a-z0-9]', '', str(value).strip(' \"\'').casefold())
ALIASES = {key(k): k for k in DEFAULT_DENSITIES}
ALIASES.update({key(k): 'Graphite' for k in ['GRA', 'graphite']})
ALIASES.update({key(k): 'Water' for k in ['H2O', 'DI water', 'deionised water', 'deionized water']})
ALIASES.update({key(k): 'NMP' for k in ['N-methyl-2-pyrrolidone', '1-methyl-2-pyrrolidone']})
def material_name(name):
    clean=str(name).strip(' \"\'')
    if key(clean) in ['', 'none', 'notused', 'nan']:
        raise ValueError('Select a material for every positive component fraction.')
    return ALIASES.get(key(clean), clean)

def density_table(overrides=None):
    result=dict(DEFAULT_DENSITIES)
    for material, value in (overrides or {}).items():
        name=material_name(material)
        try: value=float(value)
        except (TypeError,ValueError): raise ValueError('Density must be numerical for '+name) from None
        if not math.isfinite(value) or value<=0:
            raise ValueError('Density must be finite and greater than zero for '+name)
        result[name]=value
    return result

def estimate_slurry_density(values, material_densities=None):
    """Return density, component volumes and explicit assumptions, in g/cm3.

    Complete composition is required, with zero for unused slots. Solid content
    includes all five non-solvent slots. Invalid totals are never normalised.
    Unknown named materials use a visible, editable 1.6 g/cm3 assumption.
    """
    lookup={key(c): v for c,v in dict(values).items()}
    table=density_table(material_densities); parts=[]; amounts=[]
    for component, fraction, identity in SLOTS:
        if key(fraction) not in lookup:raise ValueError('Density estimation requires '+fraction+' (zero if unused).')
        try: amount=float(lookup[key(fraction)])
        except (TypeError,ValueError):raise ValueError('Component percentages must be numerical.') from None
        if not math.isfinite(amount) or not 0<=amount<=100:raise ValueError('Component percentages must be finite and between 0 and 100.')
        amounts.append(amount)
        if amount>0:
            if key(identity) not in lookup:raise ValueError('Density estimation requires '+identity)
            parts.append((component,material_name(lookup[key(identity)]),amount))
    solids=math.fsum(amounts)
    if not 0<solids<=100+1e-6:raise ValueError('Non-solvent percentages must total more than 0 and at most 100%.')
    if 'solidcontent' in lookup:
        content=float(lookup['solidcontent'])
        if not math.isfinite(content) or abs(content-solids)>1e-6:
            raise ValueError('Density estimation requires solid content to include all non-solvent components, including dispersant. Review legacy accounting first.')
    solvent=max(0.,100-solids)
    if solvent:
        if 'solvent' not in lookup:raise ValueError('Density estimation requires the solvent identity.')
        parts.append(('Solvent',material_name(lookup['solvent']),solvent))
    components=[]
    supplied={material_name(n) for n in (material_densities or {})}
    for component,material,amount in parts:
        rho=table.get(material,table[FALLBACK])
        fallback=material not in DEFAULT_DENSITIES and rho==table[FALLBACK]
        edited=material in supplied and (material not in DEFAULT_DENSITIES or rho!=DEFAULT_DENSITIES[material])
        assumption=('Fallback assumption; verify or edit this material density.' if fallback else
            'User-supplied material density; verify grade and temperature.' if edited else
            NOTES.get(material,'Reference assumption; verify the actual material grade.'))
        components.append(dict(component=component,material=material,wt_pct_slurry=amount,
            density_g_cm3=rho,specific_volume_cm3_per_g=amount/100/rho,
            assumption=assumption,fallback=fallback,edited=edited))
    density=1/math.fsum(c['specific_volume_cm3_per_g'] for c in components)
    return dict(density_g_cm3=density,formula=FORMULA,components=components,
        material_densities={c['material']:c['density_g_cm3'] for c in components})

def apply_density(values):
    """Opt-in adapter; unmarked and manual inputs retain their density values."""
    result=dict(values);mode=result.get(MODE,'manual')
    if mode not in ['manual','estimated']:raise ValueError('Choose manual or estimated slurry density.')
    if mode=='manual':return result
    record=estimate_slurry_density(result,result.get(MATERIALS))
    fields=[c for c in result if key(c)=='slurrydensitygcm3']
    if MANUAL not in result and fields:result[MANUAL]=result[fields[0]]
    for c in fields or ['Slurry density g/cm3']:result[c]=record['density_g_cm3']
    result['slurry_density_formula']=FORMULA
    return result
