import torch
# import cv2
print(torch.__version__)
print(torch.version.cuda)
print("Is CUDA available:", torch.cuda.is_available())
print("Is cuDNN enabled:", torch.backends.cudnn.is_available())
print("cuDNN version:", torch.backends.cudnn.version())

# print(cv2.__file__)
# print(cv2.cuda.getCudaEnabledDeviceCount())
# print(torch.backends.cudnn.enabled)

# import torch
# print(torch.__version__)
# print(torch.distributed.is_available())
# print(hasattr(torch.distributed, 'is_initialized'))
