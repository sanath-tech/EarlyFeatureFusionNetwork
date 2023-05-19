# Copyright (c) OpenMMLab. All rights reserved.
from .beit import BEiT
from .bisenetv1 import BiSeNetV1
from .bisenetv2 import BiSeNetV2
from .cgnet import CGNet
from .erfnet import ERFNet
from .fast_scnn import FastSCNN
from .hrnet import HRNet
from .icnet import ICNet
from .mae import MAE
from .mit import MixVisionTransformer
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1c, ResNetV1d
from .resnext import ResNeXt
from .stdc import STDCContextPathNet, STDCNet
from .swin import SwinTransformer
from .timm_backbone import TIMMBackbone
from .twins import PCPVT, SVT
from .unet import UNet
from .vit import VisionTransformer
from .fuse_mit import RGBTMixVisionTransformer
from .earlyfusion_mit import EarlyFusionMixVisionTransformer
from .double_earlyfusion_mit import DoubleEarlyFusionMixVisionTransformer
#from .small_earlyfusion_mit import SmallEarlyFusionMixVisionTransformer
from .Transformer_earlyfusion_mit import SelfAttentionEarlyFusionMixVisionTransformer
from .BP_earlyfusion_mit import BPEarlyFusionMixVisionTransformer
from .CBAM_earlyfusion_mit import CBAMEarlyFusionMixVisionTransformer
from .CBAM_featurefusion_mit import CBAMFeatureFusionMixVisionTransformer
from .CBAM_middlefusion_mit import CBAMMidFusionMixVisionTransformer
from .crossattention_mit import CrossMixVisionTransformer
from .simple_crossattention_mit import SimpleCrossMixVisionTransformer
from .new_earlyfusion_mit import SmallEarlyFusionMixVisionTransformer
from .new_withdepth_earlyfusion_mit import DepthEarlyFusionMixVisionTransformer
from .three_1x1_earlyfusion_mit import ThreeEarlyFusionMixVisionTransformer
__all__ = [
    'ResNet', 'ResNetV1c', 'ResNetV1d', 'ResNeXt', 'HRNet', 'FastSCNN',
    'ResNeSt', 'MobileNetV2', 'UNet', 'CGNet', 'MobileNetV3',
    'VisionTransformer', 'SwinTransformer', 'MixVisionTransformer',
    'BiSeNetV1', 'BiSeNetV2', 'ICNet', 'TIMMBackbone', 'ERFNet', 'PCPVT',
    'SVT', 'STDCNet', 'STDCContextPathNet', 'BEiT', 'MAE','RGBTMixVisionTransformer', 'EarlyFusionMixVisionTransformer',
    'CBAMEarlyFusionMixVisionTransformer','CBAMFeatureFusionMixVisionTransformer','CBAMMidFusionMixVisionTransformer',
    'CrossMixVisionTransformer','SmallEarlyFusionMixVisionTransformer','SelfAttentionEarlyFusionMixVisionTransformer','SimpleCrossMixVisionTransformer',
    'DepthEarlyFusionMixVisionTransformer','DoubleEarlyFusionMixVisionTransformer','ThreeEarlyFusionMixVisionTransformer'

]
