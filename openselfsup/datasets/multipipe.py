import torch
from torch.utils.data import Dataset

from openselfsup.utils import build_from_cfg

from torchvision.transforms import Compose

from .registry import DATASETS, PIPELINES
from .builder import build_datasource
from .utils import to_numpy


@DATASETS.register_module
class MultiPipeDataset(Dataset):
    """Dataset for BYOL.
    """

    def __init__(self, data_source, pipeline_list, prefetch=False):
        self.data_source = build_datasource(data_source)
        self.pipeline_list = []
        for pipeline in pipeline_list:
            pipeline = [build_from_cfg(p, PIPELINES) for p in pipeline]
            self.pipeline_list.append(Compose(pipeline))

        self.pipeline_num = len(self.pipeline_list)
        self.prefetch = prefetch

    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        img_list = [
            self.pipeline_list[_](img)
            for _ in range(self.pipeline_num)
        ]
        
        if self.prefetch:
            img_list = [
                torch.from_numpy(to_numpy(_))
                for _ in img_list
            ]

        img_list = [_.unsqueeze(0) for _ in img_list]
        img_cat = torch.cat(img_list, dim=0)
        return dict(img=img_cat)

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplemented
