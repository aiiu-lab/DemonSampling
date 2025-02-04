import os
from datetime import datetime

import fire
from image_grid import create_image_grid
from generate_abstract import DemonGenerater
from utils import from_latent_to_pil

from PIL import Image

from safety_checker import safety_check_batch


class ChooseGenerator(DemonGenerater):
    def rewards(self, pils):
        """
        Compute a reward for a list of PIL images by creating an image grid.

        Args:
            pils: List of PIL images.

        Returns:
            The generated image grid as the reward.
        """
        nsfw_checks = safety_check_batch(pils)
        new_pils = []
        for pil, nsfw in zip(pils, nsfw_checks):
            if nsfw:
                black_image = Image.new("RGB", pil.size, (0, 0, 0))
                new_pils.append(black_image)
            else:
                new_pils.append(pil)
        return create_image_grid(new_pils)


def choose_generate(
    beta=0.1,
    tau="adaptive",
    K=16,             # Number of candidate samples (was action_num)
    T=64,             # Number of time steps (was sample_step)
    demon_type="tanh",# Weighting method (was weighting)
    r_of_c="baseline",
    c_steps=20,
    ode_after=0.11,
    text="U uwawa uwa",
    cfg=2,
    seed=None,
    save_pils=True,
    experiment_directory="experiments/choose_generate",
):
    """
    Generate images using demon sampling and choose the best grid via a custom reward.

    Args:
        beta (float): Noise scaling factor.
        tau (float or str): Temperature or 'adaptive' for dynamic adjustment.
        K (int): Number of candidate samples.
        demon_type (str): Weighting method (e.g., "tanh").
        T (int): Number of time steps.
        c_steps (int): Correction steps for the baseline.
        r_of_c (str): Reward or consistency mode ("baseline" or "consistency").
        ode_after (float): Threshold for switching to ODE-based sampling.
        text (str): The text prompt for guidance.
        cfg (int or float): Guidance scale.
        seed (int): Random seed.
        save_pils (bool): Whether to save intermediate PIL images.
        experiment_directory (str): Directory for saving experiment outputs.
    """
    generator = ChooseGenerator(
        beta=beta,
        tau=tau,
        K=K,
        T=T,
        demon_type=demon_type,
        r_of_c=r_of_c,
        c_steps=c_steps,
        ode_after=ode_after,
        cfg=cfg,
        seed=seed,
        save_pils=save_pils,
        experiment_directory=experiment_directory,
    )

    generator.generate(prompt=text)


if __name__ == "__main__":
    fire.Fire(choose_generate)

# Example command to run:
# python pipelines/choose_generate.py --text "A chair made of avocado"
