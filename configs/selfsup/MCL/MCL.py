import copy
_base_ = '../../base.py'
# model settings
convert_sync_bn = True
unique_batch_size = 128
unique_world_size = 4 * 8

model = dict(
    type='MCL',
    pretrained=None,
    base_momentum=0.99,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[1,2,3,4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='DetConvGpNonLinearHead',
        in_channels=256,
        hid_channels=2048,
        out_channels=256,
        avg_size=7,
        stacked_convs=4,
        norm_cfg=dict(type='BN'),
        mlp_bn = True,
        last_bn = True,
    ),
    head=dict(
        type='LatentMultiPredictContrastHead', 
        predictor = dict(
                        type='NonLinearNeckV2',
                        in_channels=256, 
                        hid_channels=2048,
                        out_channels=256, 
                        with_avg_pool=False),
        T=1.0,
        head_num = 4,
    ),
    
    train_cfg=dict(
        downsample_ratios=[1, 2, 4, 8],
        batch_sizes = [unique_batch_size, unique_batch_size//4, unique_batch_size//16, unique_batch_size//32],
        loss_weight = [1./2, 1./4, 1./8, 1./8],
        img_pyramid_pipe_idxs_1 = [0, 1, 2, 3],
        img_pyramid_pipe_idxs_2 = [4, 5, 6, 7],
        fpn_cfg=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            norm_cfg=dict(type='BN', requires_grad=True),
            start_level=0,
            num_outs=4,
        ),
    ),
)
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=False,
    mclient_path=None)
data_train_list = 'data/imagenet/meta/train.txt'
data_train_root = 'data/imagenet/train'
dataset_type = 'MultiPipeDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(type='RandomHorizontalFlip'),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='GaussianBlur',
                sigma_min=0.1,
                sigma_max=2.0)
        ],
        p=1.),
    dict(type='RandomAppliedTrans',
         transforms=[dict(type='Solarization')], p=0.),
]
# prefetch
prefetch = True
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])


train_pipeline1 = copy.deepcopy(train_pipeline)
train_pipeline2 = copy.deepcopy(train_pipeline)
train_pipeline2[4]['p'] = 0.1 # gaussian blur
train_pipeline2[5]['p'] = 0.2 # solarization

pipeline_num = 8
train_pipeline_list = [copy.deepcopy(train_pipeline1), copy.deepcopy(train_pipeline1), copy.deepcopy(train_pipeline1), copy.deepcopy(train_pipeline1), \
    copy.deepcopy(train_pipeline2), copy.deepcopy(train_pipeline2), copy.deepcopy(train_pipeline2), copy.deepcopy(train_pipeline2)]


data = dict(
    imgs_per_gpu=unique_batch_size,
    workers_per_gpu=7,
    drop_last=True,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_source=dict(
                list_file=data_train_list, root=data_train_root,
                **data_source_cfg),
            pipeline_list=train_pipeline_list,
            prefetch=prefetch,
        )
    )
)

# additional hooks
update_interval = 1  # interval for accumulate gradient
custom_hooks = [
    dict(type='BYOLHook', end_momentum=1., update_interval=update_interval)
]

learning_rate = unique_batch_size * unique_world_size // 256 * 1.0
# optimizer
optimizer = dict(type='LARS', lr=learning_rate, weight_decay=0.00001, momentum=0.9,
                 paramwise_options={
                    '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0., lars_exclude=True),
                    'bias': dict(weight_decay=0., lars_exclude=True)})

use_fp16=True
optimizer_config = dict(grad_clip=dict(max_norm=500, norm_type=2), update_interval=update_interval, use_fp16=use_fp16)
# learning policy
lr_config = dict(
    policy='CosineAnnealingInterval',
    min_lr=0.0001,
    warmup='linear',
    warmup_iters=3127,
    warmup_ratio=0.0001, # cannot be 0
    warmup_by_epoch=False,
    by_epoch=False)

checkpoint_config = dict(interval=4)
# runtime settings
total_epochs = 100