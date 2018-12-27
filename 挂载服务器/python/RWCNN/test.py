import torch
#torch.backends.cudnn.enabled = False

x = torch.Tensor([1.0])
xx = x.cuda()
print(xx)

# CUDNN TEST
from torch.backends import cudnn
print(cudnn.is_acceptable(xx))
# for m in model.modules():
#     if isinstance(m,nn.Conv2d):
#         init.normal(m.weight.data)
#         init.xavier_normal(m.weight.data)
#         init.kaiming_normal(m.weight.data)
#         m.bias.data.fill_(0)
#     elif isinstance(m,nn.Linear):
#         m.weight.data.normal_()

