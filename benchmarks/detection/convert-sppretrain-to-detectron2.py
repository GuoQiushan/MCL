#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pickle as pkl
import sys
import torch

if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")
    obj = obj["state_dict"]
    
    include_head = True

    newmodel = {}
    for k, v in obj.items():
        old_k = k
        
        if "lateral_convs" in k or "fpn_convs" in k:
            k = k.replace('encoder_q.1.', '')
            fpn_idx = int(k.split('.')[-3])
            k = k.replace('{}.'.format(fpn_idx), '')
            if 'lateral' in k:
                k = k.replace('lateral_convs', 'fpn_lateral{}'.format(fpn_idx + 2))
                k = k.replace('conv.', '')
                k = k.replace('bn.', 'norm.')
            elif 'fpn' in k:
                k = k.replace('fpn_convs', 'fpn_output{}'.format(fpn_idx + 2))
                k = k.replace('conv.', '')
                k = k.replace('bn.', 'norm.')
        else:
            
            if 'cls_convs' in k and include_head:
                conv_id = int(k.split('.')[-3])
                k = k.replace('encoder_q.2.', '')
                k = k.replace('cls_convs', 'box_head')
                k = k.replace('{}.'.format(conv_id), 'conv{}.'.format(conv_id+1))
                k = k.replace('conv.', '')
                k = k.replace('bn.', 'norm.')
                
            elif 'shared_fc' in k and include_head:
                k = k.replace('encoder_q.2.shared_fc.0', 'box_head.fc1')
            else:
                if "layer" not in k:
                    k = "stem." + k

                for t in [1, 2, 3, 4]:
                    k = k.replace("layer{}".format(t), "res{}".format(t + 1))
                for t in [1, 2, 3]:
                    k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
                k = k.replace("downsample.0", "shortcut")
                k = k.replace("downsample.1", "shortcut.norm")
                k = k.replace("backbone.", "")
        
        print(old_k, "->", k)
        newmodel[k] = v.numpy()

    res = {
        "model": newmodel,
        "__author__": "OpenSelfSup",
        "matching_heuristics": True
    }

    assert sys.argv[2].endswith('.pkl')
    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
