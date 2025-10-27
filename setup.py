from setuptools import setup, find_packages

setup(
    name='diffusion_sde',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'':'src'},
    install_requires=[
        'accelerate',
        'diffusers',
        'image-reward',
        'hpsv2',
        'open-clip-torch',
        'openai-clip',
        'openai',
        'torch',
        'torchvision',
        'transformers',
        'huggingface_hub',
        'scipy',
        'fire'
    ],    
)
