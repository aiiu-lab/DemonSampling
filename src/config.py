import torch
DTYPE = torch.float16
FILE_PATH = "runwayml/stable-diffusion-v1-5"
C_FILE_PATH = "latent-consistency/lcm-lora-sdv1-5"
# FILE_PATH = "CompVis/stable-diffusion-v1-4"
DEVICE = torch.device("cuda")
IMAGE_DIMENSION = 512