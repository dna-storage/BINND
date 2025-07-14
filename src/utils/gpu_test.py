import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device())
if torch.cuda.is_available():
    print(f"Cuda is Availabe count={torch.cuda.device_count()}")
else:
    print("Cuda Can't be found")
