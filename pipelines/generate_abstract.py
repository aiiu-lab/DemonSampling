# Standard library imports
import json
import os
from datetime import datetime
from abc import ABC, abstractmethod
from PIL import Image

# Third-party library imports
import numpy as np
import torch
import matplotlib.pyplot as plt

# Local application/library specific imports
from api import demon_sampling, get_init_latent, odeint
from utils import from_latent_to_pil
from safety_checker import safety_check_batch


class DemonGenerater(ABC):
    """
    Base class for generating images from latent variables using a demon sampling process.
    
    Parameters:
        cfg (int or float): Guidance scale.
        beta (float): Noise scaling factor.
        tau (float or str): Temperature or 'adaptive' for dynamic adjustment.
        K (int): The number of candidate samples.
        T (int): The number of time steps.
        demon_type (str): Type of weighting to apply ('tanh', 'boltzmann', or 'optimal').
        r_of_c (str): Reward or consistency mode ("baseline" or "consistency").
        c_steps (int): Number of correction steps (meaningful when r_of_c is "baseline").
        ode_after (float): Transition threshold to ODE-based sampling.
        seed (int or None): Random seed; if None, the current timestamp is used.
        save_pils (bool): Whether to save intermediate PIL images.
        ylabel (str): Label for the y-axis when plotting rewards.
        experiment_directory (str): Directory to store experiment outputs.
    """
    def __init__(
        self,
        cfg=2,
        beta=0.1,
        tau="adaptive",
        K=16,
        T=64,
        demon_type="tanh",
        r_of_c="baseline",
        c_steps=20,
        ode_after=0.11,
        seed=None,
        save_pils=False,
        ylabel="Reward",
        experiment_directory="experiments/generate",
    ):
        self.beta = beta
        self.tau = tau
        self.K = K                    # Number of candidate samples
        self.demon_type = demon_type  # Weighting method
        self.T = T                    # Number of time steps
        self.r_of_c = r_of_c
        self.c_steps = c_steps
        self.ode_after = ode_after
        self.cfg = cfg
        self.save_pils = save_pils
        self.ylabel = ylabel
        self.experiment_directory = experiment_directory

        # Establish seed for reproducibility
        self.seed = seed if seed is not None else int(datetime.now().timestamp())
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def rewards_latent(self, latents):
        """
        Convert latents to images, optionally save them, and compute rewards.
        
        Args:
            latents: Latent representations.
            
        Returns:
            Rewards computed from the generated images.
        """
        pils = from_latent_to_pil(latents)
        if self.save_pils:
            trajectory_dir = os.path.join(self.log_dir, "trajectory")
            os.makedirs(trajectory_dir, exist_ok=True)
            nowtime = int(datetime.now().timestamp() * 1e6)
            for i, pil in enumerate(pils):
                pil.save(os.path.join(trajectory_dir, f"{nowtime}_{i}.png"))
        return self.rewards(pils)

    @abstractmethod
    def rewards(self, pils):
        """
        Abstract method to compute rewards given PIL images.
        
        Args:
            pils: A list or batch of PIL images.
            
        Returns:
            Computed reward values.
        """
        pass

    def generate_pyplot(self, log_txt, out_img_file):
        """
        Generate a plot from a log file containing rewards data.
        
        Args:
            log_txt (str): Path to the log file with reward data.
            out_img_file (str): Output image file path for the generated plot.
        """
        scores, std_devs, ts = [], [], []
        with open(log_txt, "r") as f:
            for line in f.readlines():
                score, std_dev, t, _ = map(float, line.split())
                scores.append(score)
                std_devs.append(std_dev)
                ts.append(t)
                
        plt.figure(figsize=(10, 6))
        plt.errorbar(ts, scores, yerr=std_devs, fmt='-o', capsize=5,
                     capthick=1, ecolor='red', markeredgecolor="black", color='blue')
        plt.title(f'{self.ylabel} vs Noise Level t')
        plt.xlabel('t')
        plt.ylabel(self.ylabel)
        plt.gca().invert_xaxis()  # Display larger sigmas on the left
        plt.xscale('log')
        plt.grid(True)
        plt.savefig(out_img_file)
        plt.close()

    def generate(self, prompt, ode=False):
        """
        Execute the generation process using demon sampling.
        
        Sets up the experiment directory, saves the configuration, generates an initial image,
        and then iteratively updates the latent via demon sampling.
        
        Args:
            prompt (str): The text prompt for guidance.
            ode (bool): Whether to use an ODE-based method exclusively.
        """
        # Create a unique logging directory.
        datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.join(self.experiment_directory, datetime_str)
        os.makedirs(self.log_dir, exist_ok=False)

        # Save configuration for reproducibility.
        self.config = {
            "beta": self.beta,
            "tau": self.tau,
            "K": self.K,
            "T": self.T,
            "demon_type": self.demon_type,
            "r_of_c": self.r_of_c,
            "c_steps": self.c_steps,
            "ode_after": self.ode_after,
            "cfg": self.cfg,
            "prompt": prompt,
            "seed": self.seed,
            "log_dir": self.log_dir,
        }
        with open(os.path.join(self.log_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)
        
        # Generate and save the initial image.
        latent = get_init_latent()
        init_prompts = {"prompts": [prompt if prompt is not None else ""], "cfgs": [self.cfg]}
        initial_latent = odeint(latent, init_prompts, self.T)
        from_latent_to_pil(initial_latent).save(os.path.join(self.log_dir, 'init.png'))

        # If not using exclusive ODE mode, apply demon sampling.
        if not ode:
            latent = demon_sampling(
                latent,
                init_prompts,
                self.rewards_latent,  # Reward function receives latents.
                self.beta,
                self.tau,
                self.K,
                self.T,
                demon_type=self.demon_type,
                r_of_c=self.r_of_c,
                c_steps=self.c_steps,
                ode_after=self.ode_after,
                log_dir=self.log_dir,
            )
            pil = from_latent_to_pil(latent)
            nsfw_checks = safety_check_batch([pil])[0]
            
            if nsfw_checks:
                pil = Image.new("RGB", pil.size, (0, 0, 0))
                print("NSFW image detected. Replaced with a black image.")

            pil.save(os.path.join(self.log_dir, 'out.png'))
            self.generate_pyplot(
                os.path.join(self.log_dir, "expected_reward.txt"),
                os.path.join(self.log_dir, "expected_reward.png")
            )
