import hydra
import torch

from mask3d.models.mask3d import Mask3D
from mask3d.utils.utils import (
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)

class InstanceSegmentation(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = hydra.utils.instantiate(cfg.model)


    def forward(self, x, raw_coordinates=None):
        return self.model(x, raw_coordinates=raw_coordinates)
    

from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.experimental import initialize, compose

def get_model():


  # Initialize the directory with config files
  with initialize(config_path="conf"):
      # Compose a configuration
      cfg = compose(config_name="config_base_instance_segmentation.yaml")
      print(OmegaConf.to_yaml(cfg))

  # # Initialize the Hydra context
  # hydra.core.global_hydra.GlobalHydra.instance().clear()
  # hydra.initialize(config_path="conf")

  # Load the configuration
  # cfg = hydra.compose(config_name="config_base_instance_segmentation.yaml")
  model = InstanceSegmentation(cfg)

  if cfg.general.backbone_checkpoint is not None:
      cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(
          cfg, model
      )
  if cfg.general.checkpoint is not None:
      cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

  return model
