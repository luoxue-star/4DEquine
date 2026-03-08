from .vit import vith, vitl, vits
from .encoders.dinov2_fusion_wrapper import Dinov2FusionWrapper
from .encoders.dinov3_fusion_warpper import Dinov3FusionWrapper
from torch import nn
import torchvision

def create_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE == 'vith':
        return vith(cfg)
    elif cfg.MODEL.BACKBONE.TYPE == 'vitl':
        return vitl(cfg)
    elif cfg.MODEL.BACKBONE.TYPE == 'vits':
        return vits(cfg)
    elif cfg.MODEL.BACKBONE.TYPE == 'dinov2_fusion':
        return Dinov2FusionWrapper(cfg)
    elif cfg.MODEL.BACKBONE.TYPE == 'dinov3_fusion':
        return Dinov3FusionWrapper(cfg)
    else:
        raise NotImplementedError('Backbone type is not implemented')
