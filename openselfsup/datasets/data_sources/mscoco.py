import os
from PIL import Image

from ..registry import DATASOURCES
from .utils import McLoader
import json

@DATASOURCES.register_module
class Mscoco(object):

    def __init__(self, root, list_file, memcached=False, mclient_path=None, return_label=True):
        with open(list_file, 'r') as f:
            anno = json.load(f)
        
        image_list =[ _['file_name'] for _ in  anno['images'] ]
        self.fns = [os.path.join(root, fn) for fn in image_list]
        self.memcached = memcached
        self.mclient_path = mclient_path
        self.initialized = False
        self.has_labels = False
        self.return_label = False

    def _init_memcached(self):
        if not self.initialized:
            assert self.mclient_path is not None
            self.mc_loader = McLoader(self.mclient_path)
            self.initialized = True

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        if self.memcached:
            self._init_memcached()
        if self.memcached:
            img = self.mc_loader(self.fns[idx])
        else:
            img = Image.open(self.fns[idx])
        img = img.convert('RGB')
        if self.has_labels and self.return_label:
            target = self.labels[idx]
            return img, target
        else:
            return img
