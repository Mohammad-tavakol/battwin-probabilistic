"""Small application service layer suitable for CLI, notebooks or a later GUI."""
import json
from pathlib import Path
from .config import project_root,stage_schema,TrainingConfig,TuningConfig,SimulationConfig
from .data import load_stage
from .models import ModelCatalog
from .chain import ProcessChain

class Project:
    def __init__(self,path=None):
        self.root=project_root(path)
        self.catalog=ModelCatalog(project=self.root)

    def schema(self,stage):return stage_schema(stage,self.root)

    def data(self,stage):return load_stage(stage,self.root)

    def model(self,stage,calibration='kuleshov'):return self.catalog.load(stage,calibration)

    def chain(self,config=None):return ProcessChain(self.catalog,config or SimulationConfig())

    def train(self,stage,output_dir,training=None,tuning=None,use_selected=True,callback=None):
        from .training import train_process,fit_process
        data=self.data(stage);schema=data['schema'];cfg=training or TrainingConfig()
        if use_selected:
            path=Path(output_dir)/f'{stage}.joblib'
            return train_process(data['train'],data['validation'],schema,schema['target_configurations'],cfg,path,callback)
        artifact,_=fit_process(data['development'],schema,cfg,tuning or TuningConfig(),output_dir=output_dir,callback=callback)
        return artifact
