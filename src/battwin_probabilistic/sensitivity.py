"""Simulation-only PFI for fixed-population propagation; no measured targets required."""
from .orchestration.core import compute_chain_pfi as _compute_chain_pfi,summarise_chain_pfi,select_pfi_features
from .formulation import SOLIDS,key as formulation_key

def compute_chain_pfi(line,df_input,features=None,feature_groups=None,*args,**kwargs):
    """Sensitivity metric over independent controls; solids follow fractions."""
    is_solids=lambda c:formulation_key(c)==formulation_key(SOLIDS)
    if features is not None and any(is_solids(c) for c in features):
        raise ValueError('Solid content is derived. Permute its component fractions instead.')
    if feature_groups and any(is_solids(c) for columns in feature_groups.values() for c in columns):
        raise ValueError('Solid content is derived and cannot be an independent PFI group member.')
    features=[c for c in select_pfi_features(df_input,line,features) if not is_solids(c)]
    return _compute_chain_pfi(line,df_input,features,feature_groups,*args,**kwargs)
