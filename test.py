import torch
print(torch.version.cuda)
print(torch.__version__)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
