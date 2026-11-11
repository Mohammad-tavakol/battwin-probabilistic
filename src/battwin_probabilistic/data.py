"""Condition-safe data handling. Repetitions are retained, never averaged away."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from .config import project_root,stage_schema

def read_table(path):
    p=Path(path)
    return pd.read_csv(p,float_precision='round_trip') if p.suffix.lower()=='.csv' else pd.read_excel(p)

def condition_id(frame,columns):
    values=frame[list(columns)].copy()
    for col in columns:
        if pd.api.types.is_numeric_dtype(values[col]):values[col]=values[col].astype(float).replace(-0.,0.)
        else:values[col]=values[col].astype('string').fillna('<NA>')
    return pd.util.hash_pandas_object(values,index=False).map(lambda v:f'{v:016x}')

def annotate(frame,schema):
    frame=frame.copy()
    required=schema['features']+schema['targets']
    missing=set(required)-set(frame)
    if missing:raise ValueError(f'Missing model inputs/targets: {sorted(missing)}')
    frame['condition_id']=condition_id(frame,schema['features'])
    frame['group_id']=condition_id(frame,schema['split_group_features'])
    if '_source_row' not in frame:frame['_source_row']=np.arange(len(frame))
    return frame

def assert_disjoint(*frames):
    for i,a in enumerate(frames):
        for b in frames[i+1:]:
            for key in ['condition_id','group_id']:
                if key in a and key in b and set(a[key])&set(b[key]):raise ValueError(f'Partition leakage: {key}')

def grouped_split(frame,fraction=.2,seed=42,group_column='group_id'):
    if not 0<fraction<1:raise ValueError('Split fraction must be between 0 and 1')
    if frame[group_column].nunique()<3:raise ValueError('At least three groups are required')
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=fraction,random_state=seed).split(frame,groups=frame[group_column]))
    train,validation=frame.iloc[a].copy(),frame.iloc[b].copy()
    if group_column=='group_id':assert_disjoint(train,validation)
    elif set(train.condition_id)&set(validation.condition_id):raise ValueError('Condition leakage')
    return train,validation

def load_stage(stage,project=None):
    root=project_root(project);schema=stage_schema(stage,root)
    result={name:annotate(read_table(root/'data/stages'/stage/f'{name}.csv'),schema) for name in ['train','validation','test']}
    assert_disjoint(*result.values())
    result['development']=pd.concat([result['train'],result['validation']],ignore_index=True)
    result['schema']=schema
    return result
