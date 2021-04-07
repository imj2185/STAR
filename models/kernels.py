from .layers import _Swish
# import torch.nn.functional as fn
from fast_transformers.feature_maps import ActivationFunctionFeatureMap


swish_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: _Swish.apply(x) + 1
    # lambda x: fn.silu(x) + 1
)
