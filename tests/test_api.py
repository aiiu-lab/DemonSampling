import os
from PIL import Image


from api import odeint, sdeint, get_init_latent, latent_sde, oderevert
from utils import get_condition, from_latent_to_pil, from_pil_to_latent

if not os.path.exists('./tmp'):
    os.makedirs('./tmp')



def test_odeint():
    condition = {
        "prompts": ["On Moon", "Astronaut", "Riding a donkey"],
        "cfgs": [3, 2, 4]
    }
    steps = 20
    x = get_init_latent() # sigma is 14.6488 for stable diffusion
    x = odeint(x, condition, steps)
    pil = from_latent_to_pil(x)
    pil.save('./tmp/odeint.png')
    assert os.path.exists('./tmp/odeint.png')

def test_oderevert():
    condition = {
        "prompts": ["An astronaut riding a horse on Mars."],
        "cfgs": [5]
    }
    pil = Image.open("assets/test_image.png")
    x = from_pil_to_latent(pil)
    x = oderevert(x, condition)
    x = odeint(x, condition, 20)
    pil = from_latent_to_pil(x)
    pil.save('./tmp/oderevert.png')
    assert os.path.exists('./tmp/oderevert.png')

def test_sdedit():
    old_condition = condition = {
        "prompts": ["An astronaut riding a horse on Mars."],
        "cfgs": [5]
    }
    new_condition = {
        "prompts": ["On Moon", "Astronaut", "Riding a donkey"],
        "cfgs": [3, 2, 4]
    }
    steps = 20
    sigma = 14
    beta = 0.125
    pil = Image.open("assets/test_image.png")
    x = from_pil_to_latent(pil)
    x = oderevert(x, condition, start_t=sigma)
    
    x = sdeint(x, new_condition, beta, steps, start_t=sigma)
    pil = from_latent_to_pil(x)
    pil.save('./tmp/sdedit.png')


# Run the following command to test the function:
# time pytest tests/test_api.py