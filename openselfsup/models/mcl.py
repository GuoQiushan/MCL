import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS
import torch.nn.functional as F

@MODELS.register_module
class MCL(nn.Module):
    """MCL.
    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        momentum (float): Momentum coefficient for the momentum-updated encoder.
            Default: 0.999.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 base_momentum=0.996,
                 train_cfg = None,
                 **kwargs):
        super(MCL, self).__init__()

        self.train_cfg = train_cfg
        fpn_cfg = train_cfg.get('fpn_cfg')

        self.encoder_q = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(fpn_cfg), builder.build_neck(neck))
        self.encoder_k = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(fpn_cfg), builder.build_neck(neck))
        self.backbone = self.encoder_q[0]
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        self.head = builder.build_head(head)
        self.init_weights(pretrained=pretrained)

        self.base_momentum = base_momentum
        self.momentum = base_momentum

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.encoder_q[0].init_weights(pretrained=pretrained)
        self.encoder_q[1].init_weights()
        self.encoder_q[2].init_weights(init_linear='kaiming')
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
        
        # init the predictor in the head
        self.head.init_weights()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def momentum_update(self):
        self._momentum_update_key_encoder()


    @torch.no_grad()
    def random_combine_img(self, img, batch_size, downsample_ratio=2):
        num_per_img = int(downsample_ratio ** 2)
        n_h, n_w = downsample_ratio, downsample_ratio

        num_img, C = img.shape[0], img.shape[1]
        random_idx = torch.randperm(num_per_img * batch_size, device = img.device)
        random_idx = torch.fmod(random_idx, num_img).long()
        img_downsampled = F.interpolate(img, scale_factor = 1. / downsample_ratio)

        img_downsampled = img_downsampled.type(img.type())
        img_ds = img_downsampled[random_idx]

        H, W = img_ds.shape[2], img_ds.shape[3]
        img_ds = img_ds.reshape(batch_size, n_h, n_w, C, H, W)
        img_ds = img_ds.permute(0, 3, 1, 4, 2, 5).contiguous()
        img_ds = img_ds.reshape(batch_size, C, n_h*H, n_w*W)

        gt_label = torch.arange(num_img, device=img.device)
        gt_label_ds = gt_label[random_idx]
        return img_ds, gt_label_ds

    @torch.no_grad()
    def generate_img_pyramid(self, img):
        num_pipeline = img.shape[1]
        img_list = [img[:, _, ...].contiguous() for _ in range(num_pipeline)]
        img_list_1 = img_list
        img_list_2 = img_list
        num_pipeline_1 = len(img_list_1)
        num_pipeline_2 = len(img_list_2)

        downsample_ratios = self.train_cfg.get('downsample_ratios', [1, 2, 4])
        batch_sizes = self.train_cfg.get('batch_sizes', [32, 16, 16])
        level_num = len(downsample_ratios)

        img_qs, gt_qs = [], []
        img_ks, gt_ks = [], []

        pipe_idxs_1 = self.train_cfg.get('img_pyramid_pipe_idxs_1', [0, 0, 0, 0])
        pipe_idxs_2 = self.train_cfg.get('img_pyramid_pipe_idxs_2', [1, 1, 1, 1])

        for i in range(level_num):
            if i == 0:
                img_qs.append(img_list_1[pipe_idxs_1[i]])
                gt_qs.append(torch.arange(img.shape[0], device=img.device))
                img_ks.append(img_list_2[pipe_idxs_2[i]])
                gt_ks.append(torch.arange(img.shape[0], device=img.device))
            else:
                img_to_combine = img_list_1[pipe_idxs_1[i]]
                _img, _gt = self.random_combine_img(img_to_combine,
                        batch_sizes[i], downsample_ratios[i])
                img_qs.append(_img)
                gt_qs.append(_gt)

                img_to_combine = img_list_2[pipe_idxs_2[i]]
                _img, _gt = self.random_combine_img(img_to_combine,
                        batch_sizes[i], downsample_ratios[i])
                img_ks.append(_img)
                gt_ks.append(_gt)

        img_num = torch.tensor([_.shape[0] for _ in img_qs], device=img.device)
        img_num_idx = torch.cumsum(img_num, 0)
        img_qs = torch.cat(img_qs)
        img_ks = torch.cat(img_ks)

        return img_qs, img_ks, img_num_idx, gt_qs, gt_ks

    def get_loss_suffix(self, loss, suffix = '_level_1'):
        loss_suffix = {}
        for key in loss.keys():
            key_suffix = key + suffix
            if isinstance(loss[key], dict):
                loss_suffix[key_suffix] = {}
                for inner_key in loss[key]:
                    inner_key_suffix = inner_key + suffix
                    loss_suffix[key_suffix][inner_key_suffix] = loss[key][inner_key]
            else:    
                loss_suffix[key_suffix] = loss[key]
        return loss_suffix

    def forward_train(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        
        img_qs, img_ks, img_num_idx, gt_qs, gt_ks = self.generate_img_pyramid(img)

        # feed concated image pyramid to FPN encoder and get output vectors for all levels
        qs_vectors_online = self.encoder_q(img_qs)  
        ks_vectors_online = self.encoder_q(img_ks)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            qs_vectors_target = self.encoder_k(img_qs)
            ks_vectors_target = self.encoder_k(img_ks)

        # set highest level features as target and normalize them
        q_target = qs_vectors_target[-1][:img_num_idx[0]] 
        q_target = q_target.reshape(-1, q_target.shape[-1])
        q_target = nn.functional.normalize(q_target, dim=1)

        k_target = ks_vectors_target[-1][:img_num_idx[0]] # highest level feature
        k_target = k_target.reshape(-1, k_target.shape[-1])
        k_target = nn.functional.normalize(k_target, dim=1)

        # concat all target vectors
        q_target_all = concat_all_gather(q_target)
        k_target_all = concat_all_gather(k_target)

        loss_weight = self.train_cfg.get('loss_weight', [1, 1, 2])
        losses = dict()
        losses['momentum'] = torch.tensor(self.momentum, device = q_target.device)
        level_num = len(gt_qs)

        # the num of unique image 
        unique_batch_size = img_num_idx[0]

        for i in range(level_num):
            fpn_idx = -(i+1)
            start_idx = 0 if i == 0 else img_num_idx[i-1]
            end_idx = img_num_idx[i]

            # select fpn level and pick out the corresponding feature batch
            q_online = qs_vectors_online[fpn_idx]
            q_online = q_online[start_idx:end_idx]
            # reshape online query to 2d for mlp
            q_online = q_online.reshape(-1, q_online.shape[-1])
            
            # select fpn level and pick out the corresponding feature batch
            k_online = ks_vectors_online[fpn_idx]
            k_online = k_online[start_idx:end_idx]
            # reshape online query to 2d for mlp
            k_online = k_online.reshape(-1, k_online.shape[-1])

            # align the gt with concated target vectors
            aligned_gt_qs = gt_qs[i] + unique_batch_size * torch.distributed.get_rank()
            aligned_gt_ks = gt_ks[i] + unique_batch_size * torch.distributed.get_rank()
            # predict in head
            _losses_q = self.head(q_online, k_target_all, aligned_gt_qs, loss_weight[i], i)
            _losses_k = self.head(k_online, q_target_all, aligned_gt_ks, loss_weight[i], i)

            loss_suffix_q = self.get_loss_suffix(_losses_q, '_q_level_{}'.format(i))
            loss_suffix_k = self.get_loss_suffix(_losses_k, '_k_level_{}'.format(i))
            losses.update(loss_suffix_q)
            losses.update(loss_suffix_k)


        return losses

    def forward_test(self, img, **kwargs):
        pass

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.

    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
