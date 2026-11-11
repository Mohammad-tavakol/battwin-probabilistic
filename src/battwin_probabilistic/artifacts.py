"""Single-file process artifacts and filesystem-independent loading."""
from pathlib import Path
import os,hashlib
from joblib import load,dump

ARTIFACT_TYPE='xgboostlss_kuleshov_process'

def native_path(path):
    p=Path(path).expanduser().resolve()
    return Path('\\\\?\\'+str(p)) if os.name=='nt' and not str(p).startswith('\\\\?\\') else p

def sha256(path):
    h=hashlib.sha256()
    with native_path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()

def load_artifact(path):
    artifact=load(native_path(path))
    if artifact.get('artifact_type')!=ARTIFACT_TYPE or artifact.get('format_version')!=1:
        raise ValueError('Expected a combined BATTWIN Kuleshov process artifact, format 1')
    bundle=artifact['model_bundle'];maps=artifact['kuleshov_calibration']
    if set(bundle['target_ensembles'])!=set(maps):raise ValueError('Model/map targets differ')
    for target,record in maps.items():
        expected=artifact.get('provenance',{}).get('source_model_sha256')
        if expected and record.get('base_bundle_sha256')!=expected:raise ValueError('Map/model identity mismatch')
    return artifact

def load_base_bundle(path):
    artifact=load(native_path(path))
    return artifact['model_bundle'] if artifact.get('artifact_type')==ARTIFACT_TYPE else artifact

def save_artifact(artifact,path):
    if artifact.get('artifact_type')!=ARTIFACT_TYPE:raise ValueError('Unknown artifact format')
    path=native_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    dump(artifact,temporary,compress=3)
    os.replace(temporary,path)
    return path
