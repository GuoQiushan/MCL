import torch
import torch.nn as nn
from packaging import version
from mmcv.cnn import kaiming_init, normal_init

from .registry import NECKS
from .utils import build_norm_layer

import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from torch.nn.modules.batchnorm import _BatchNorm

def _init_weights(module, init_linear='normal', std=0.01, bias=0.):
    assert init_linear in ['normal', 'kaiming'], \
        "Undefined init_linear: {}".format(init_linear)
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if init_linear == 'normal':
                normal_init(m, std=std, bias=bias)
            else:
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                            nn.GroupNorm, nn.SyncBatchNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


@NECKS.register_module
class LinearNeck(nn.Module):
    """Linear neck: fc only.
    """

    def __init__(self, in_channels, out_channels, with_avg_pool=True):
        super(LinearNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, out_channels)

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [self.fc(x.view(x.size(0), -1))]


@NECKS.register_module
class RelativeLocNeck(nn.Module):
    """Relative patch location neck: fc-bn-relu-dropout.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 sync_bn=False,
                 with_avg_pool=True):
        super(RelativeLocNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if version.parse(torch.__version__) < version.parse("1.4.0"):
            self.expand_for_syncbn = True
        else:
            self.expand_for_syncbn = False

        self.fc = nn.Linear(in_channels * 2, out_channels)
        if sync_bn:
            _, self.bn = build_norm_layer(
                dict(type='SyncBN', momentum=0.003),
                out_channels)
        else:
            self.bn = nn.BatchNorm1d(
                out_channels, momentum=0.003)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()
        self.sync_bn = sync_bn

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear, std=0.005, bias=0.1)

    def _forward_syncbn(self, module, x):
        assert x.dim() == 2
        if self.expand_for_syncbn:
            x = module(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        else:
            x = module(x)
        return x

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.sync_bn:
            x = self._forward_syncbn(self.bn, x)
        else:
            x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return [x]


@NECKS.register_module
class NonLinearNeckV0(nn.Module):
    """The non-linear neck in ODC, fc-bn-relu-dropout-fc-relu.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 sync_bn=False,
                 with_avg_pool=True):
        super(NonLinearNeckV0, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if version.parse(torch.__version__) < version.parse("1.4.0"):
            self.expand_for_syncbn = True
        else:
            self.expand_for_syncbn = False

        self.fc0 = nn.Linear(in_channels, hid_channels)
        if sync_bn:
            _, self.bn0 = build_norm_layer(
                dict(type='SyncBN', momentum=0.001, affine=False),
                hid_channels)
        else:
            self.bn0 = nn.BatchNorm1d(
                hid_channels, momentum=0.001, affine=False)

        self.fc1 = nn.Linear(hid_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()
        self.sync_bn = sync_bn

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def _forward_syncbn(self, module, x):
        assert x.dim() == 2
        if self.expand_for_syncbn:
            x = module(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        else:
            x = module(x)
        return x

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        if self.sync_bn:
            x = self._forward_syncbn(self.bn0, x)
        else:
            x = self.bn0(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.relu(x)
        return [x]


@NECKS.register_module
class NonLinearNeckV1(nn.Module):
    """The non-linear neck in MoCo v2: fc-relu-fc.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True):
        super(NonLinearNeckV1, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [self.mlp(x.view(x.size(0), -1))]


@NECKS.register_module
class NonLinearNeckV2(nn.Module):
    """The non-linear neck in byol: fc-bn-relu-fc.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True):
        super(NonLinearNeckV2, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.BatchNorm1d(hid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1, "Got: {}".format(len(x))
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [self.mlp(x.view(x.size(0), -1))]

@NECKS.register_module
class NonLinearNeckV3(nn.Module):
    """The non-linear neck in byol: fc-bn-relu-fc-bn.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True):
        super(NonLinearNeckV3, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.BatchNorm1d(hid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels),
            nn.BatchNorm1d(out_channels, affine=False)
        )

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1, "Got: {}".format(len(x))
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [self.mlp(x.view(x.size(0), -1))]

@NECKS.register_module
class NonLinearNeckSimCLR(nn.Module):
    """SimCLR non-linear neck.

    Structure: fc(no_bias)-bn(has_bias)-[relu-fc(no_bias)-bn(no_bias)].
        The substructures in [] can be repeated. For the SimCLR default setting,
        the repeat time is 1.
    However, PyTorch does not support to specify (weight=True, bias=False).
        It only support \"affine\" including the weight and bias. Hence, the
        second BatchNorm has bias in this implementation. This is different from
        the official implementation of SimCLR.
    Since SyncBatchNorm in pytorch<1.4.0 does not support 2D input, the input is
        expanded to 4D with shape: (N,C,1,1). Not sure if this workaround
        has no bugs. See the pull request here:
        https://github.com/pytorch/pytorch/pull/29626.

    Args:
        num_layers (int): Number of fc layers, it is 2 in the SimCLR default setting.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_layers=2,
                 sync_bn=True,
                 with_bias=False,
                 with_last_bn=True,
                 with_avg_pool=True):
        super(NonLinearNeckSimCLR, self).__init__()
        self.sync_bn = sync_bn
        self.with_last_bn = with_last_bn
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if version.parse(torch.__version__) < version.parse("1.4.0"):
            self.expand_for_syncbn = True
        else:
            self.expand_for_syncbn = False

        self.relu = nn.ReLU(inplace=True)
        self.fc0 = nn.Linear(in_channels, hid_channels, bias=with_bias)
        if sync_bn:
            _, self.bn0 = build_norm_layer(
                dict(type='SyncBN'), hid_channels)
        else:
            self.bn0 = nn.BatchNorm1d(hid_channels)

        self.fc_names = []
        self.bn_names = []
        for i in range(1, num_layers):
            this_channels = out_channels if i == num_layers - 1 \
                else hid_channels
            self.add_module(
                "fc{}".format(i),
                nn.Linear(hid_channels, this_channels, bias=with_bias))
            self.fc_names.append("fc{}".format(i))
            if i != num_layers - 1 or self.with_last_bn:
                if sync_bn:
                    self.add_module(
                        "bn{}".format(i),
                        build_norm_layer(dict(type='SyncBN'), this_channels)[1])
                else:
                    self.add_module(
                        "bn{}".format(i),
                        nn.BatchNorm1d(this_channels))
                self.bn_names.append("bn{}".format(i))
            else:
                self.bn_names.append(None)

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def _forward_syncbn(self, module, x):
        assert x.dim() == 2
        if self.expand_for_syncbn:
            x = module(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        else:
            x = module(x)
        return x

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        if self.sync_bn:
            x = self._forward_syncbn(self.bn0, x)
        else:
            x = self.bn0(x)
        for fc_name, bn_name in zip(self.fc_names, self.bn_names):
            fc = getattr(self, fc_name)
            x = self.relu(x)
            x = fc(x)
            if bn_name is not None:
                bn = getattr(self, bn_name)
                if self.sync_bn:
                    x = self._forward_syncbn(bn, x)
                else:
                    x = bn(x)
        return [x]

@NECKS.register_module
class DetConvGpNonLinearHead(nn.Module):
    """The non-linear neck.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 fc_dim = 1024,
                 avg_size = 7,
                 stacked_convs = 1,
                 norm_cfg = None,
                 mlp_bn = False,
                 last_bn = False,
                 with_fc = False,
                 ):
        super(DetConvGpNonLinearHead, self).__init__()

        self.stacked_convs = stacked_convs
        self.avg_size = avg_size
        self.norm_cfg = norm_cfg
        self.mlp_bn = mlp_bn
        self.last_bn = last_bn
        self.fc_dim = fc_dim
        self.with_fc = with_fc

        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            self.cls_convs.append(
                ConvModule(
                    in_channels,
                    in_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=None,
                    norm_cfg=self.norm_cfg))
            
        self.avg_pool = nn.AvgPool2d(self.avg_size, stride=self.avg_size)
        if self.with_fc:
            self.shared_fc = nn.Sequential(nn.Linear(in_channels * self.avg_size * self.avg_size, self.fc_dim), 
                nn.BatchNorm1d(self.fc_dim), nn.ReLU())
            in_channels = self.fc_dim
            
        if self.mlp_bn == True:
            mlp = [
                nn.Linear(in_channels, hid_channels), 
                nn.BatchNorm1d(hid_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hid_channels, out_channels)
            ]
        else:
            mlp = [
                nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
                nn.Linear(hid_channels, out_channels)
            ]
        if self.last_bn == True:
            mlp.append(nn.BatchNorm1d(out_channels, affine=False))

        self.mlp = nn.Sequential(*mlp)

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        # input: multi-level 2d feature
        # output: multi-level embedding: N, nh * nw, feat_dim
        level_num = len(x)
        feats = []

        for i in range(level_num):
            cls_feat = x[i]
            
            batch_size, c, n_h_avg_size, n_w_avg_size = cls_feat.shape[0], cls_feat.shape[1], cls_feat.shape[2], cls_feat.shape[3]
            n_h, n_w = n_h_avg_size // self.avg_size, n_w_avg_size // self.avg_size
            cls_feat = cls_feat.reshape(batch_size, c, n_h, self.avg_size, n_w, self.avg_size)
            cls_feat = cls_feat.permute(0, 2, 4, 1, 3, 5).reshape(-1, c, self.avg_size, self.avg_size)

            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)

            if self.with_fc:
                cls_feat = cls_feat.reshape(cls_feat.shape[0], -1)
                cls_feat = self.shared_fc(cls_feat)
            else:
                cls_feat = self.avg_pool(cls_feat).reshape(-1, c)
            #batch_size, c, n_h, n_w = cls_feat.shape[0], cls_feat.shape[1], cls_feat.shape[2], cls_feat.shape[3]
            #cls_feat = cls_feat.permute(0,2,3,1).reshape(-1, c)
            feat = self.mlp(cls_feat)
            feat = feat.reshape(batch_size, n_h * n_w, -1)
            feats.append(feat)

        return feats

    def forward_list(self, x, img_num_idx):
        # input: multi-level 2d feature
        # output: multi-level embedding: N, nh * nw, feat_dim
        # img_num_idx: cumsum of N_224, N_112, N_56, N_28
        level_num = len(x)
        feats = []

        for i in range(level_num):
            fpn_idx = -(i+1)
            start_idx = 0 if i == 0 else img_num_idx[i-1]
            end_idx = img_num_idx[i]

            cls_feat = x[fpn_idx]
            cls_feat = cls_feat[start_idx:end_idx]

            batch_size, c, n_h_avg_size, n_w_avg_size = cls_feat.shape[0], cls_feat.shape[1], cls_feat.shape[2], cls_feat.shape[3]
            n_h, n_w = n_h_avg_size // self.avg_size, n_w_avg_size // self.avg_size
            cls_feat = cls_feat.reshape(batch_size, c, n_h, self.avg_size, n_w, self.avg_size)
            cls_feat = cls_feat.permute(0, 2, 4, 1, 3, 5).reshape(-1, c, self.avg_size, self.avg_size)

            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)

            if self.with_fc:
                cls_feat = cls_feat.reshape(cls_feat.shape[0], -1)
                cls_feat = self.shared_fc(cls_feat)
            else:
                cls_feat = self.avg_pool(cls_feat).reshape(-1, c)
            #batch_size, c, n_h, n_w = cls_feat.shape[0], cls_feat.shape[1], cls_feat.shape[2], cls_feat.shape[3]
            #cls_feat = cls_feat.permute(0,2,3,1).reshape(-1, c)
            feat = self.mlp(cls_feat)
            feat = feat.reshape(batch_size, n_h * n_w, -1)
            feats.append(feat)

        return feats

@NECKS.register_module
class AvgPoolNeck(nn.Module):
    """Average pooling neck.
    """

    def __init__(self):
        super(AvgPoolNeck, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self, **kwargs):
        pass

    def forward(self, x):
        assert len(x) == 1
        return [self.avg_pool(x[0])]

@NECKS.register_module
class FPN(nn.Module):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 norm_eval=False,
                 upsample_cfg=dict(mode='nearest')):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.norm_eval = norm_eval

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
    
    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(FPN, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

@NECKS.register_module
class FPNLinear(nn.Module):
    def __init__(self,
                 num_outs,
                 in_channels = 2048,
                 out_channels = 256,
                 norm_cfg = None,
                 upsample_cfg=dict(mode='nearest')):
        super(FPNLinear, self).__init__()
        self.num_outs = num_outs
        self.upsample_cfg = upsample_cfg
        self.reduction = ConvModule(
            in_channels,
            out_channels,
            1,
            stride=1,
            padding=0,
            conv_cfg=None,
            norm_cfg=norm_cfg)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
    
    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(FPNLinear, self).train(mode)
        

    def forward(self, inputs):
        """Forward function."""
        # inputs is a list
        assert len(inputs) == 1
        input_feat = self.reduction(inputs[0])
        outs = [input_feat]
        for i in range(self.num_outs - 1):
            input_feat_scale = F.interpolate(input_feat, scale_factor=2**(i+1), **self.upsample_cfg)
            outs.append(input_feat_scale)
        outs = outs[::-1]
        return tuple(outs)
        
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 output_idx=0,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 norm_eval=False,
                 upsample_cfg=dict(mode='nearest')):
        super(FPNIDX, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.norm_eval = norm_eval
        self.output_idx = output_idx

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            
        fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
        self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
    
    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(FPNIDX, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[0](laterals[self.output_idx])
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)