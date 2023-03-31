import torch
import sys
m = sys.argv[1]
model = torch.load(m, torch.device('cpu'))
model.pop('optimizer')
torch.save(model, sys.argv[2])
