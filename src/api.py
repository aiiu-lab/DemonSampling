import copy
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from karras import LatentSDEModel, LatentConsistencyModel
from utils import get_condition
from config import DEVICE, DTYPE

# Initialize global models.
latent_sde = LatentSDEModel(beta=0)
latent_consistency = LatentConsistencyModel()


class OdeModeContextManager:
    """Context manager to switch latent_sde to ODE mode during execution."""
    def __enter__(self):
        latent_sde.ode_mode()

    def __exit__(self, exc_type, exc_val, exc_tb):
        latent_sde.ode_mode_revert()

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


def duplicate_condition(conds, n):
    """Repeat each tensor in a condition dictionary n times."""
    return {
        "encoder_hidden_states": conds["encoder_hidden_states"].repeat(n, 1, 1), 
    }


def _get_f_g(t, x, prompts):
    """
    Compute the drift (f) and diffusion (g) from latent_sde.

    For each batch element, a weighted sum over multiple guidance scales is computed.
    """
    conds = prompts['conditions']
    cfgs = prompts['cfgs']
    N = x.shape[0]
    C = len(cfgs) + 1
    # If multiple inputs are provided, duplicate the condition accordingly.
    if N > 1:
        conds = duplicate_condition(conds, N)
    # Evaluate latent_sde on an interleaved batch.
    fs, g = latent_sde(t, x.repeat_interleave(C, dim=0), conds)
    all_f = [
        fs[(j + 1) * C - 1] +
        sum((fs[j * C + i] - fs[(j + 1) * C - 1]) * cfg for i, cfg in enumerate(cfgs))
        for j in range(N)
    ]
    return torch.stack(all_f), g


@torch.inference_mode()
def get_f_g(t, x, prompts):
    """
    Evaluate _get_f_g on x in chunks to avoid memory issues.

    Returns:
        (Tensor, Tensor): The drift values (f) and diffusion coefficient (g).
    """
    MAX_CHUNK_SIZE = 8
    all_fs = []
    N = x.shape[0]
    for i in range(0, N, MAX_CHUNK_SIZE):
        chunk = x[i:min(i + MAX_CHUNK_SIZE, N)]
        fs, g = _get_f_g(t, chunk, prompts)
        all_fs.append(fs)
    return torch.cat(all_fs), g


@torch.inference_mode()
def sde_step(x, t, prev_t, prompts, z):
    """
    Take a single SDE step using a Heun (predictor-corrector) scheme.
    
    Note: Expects x to have batch size 1.
    """
    assert x.shape[0] == 1, "sde_step expects a batch size of 1."
    dt = t - prev_t
    rand_term = z * torch.sqrt(torch.abs(dt))
    f1, g1 = get_f_g(prev_t, x, prompts)
    x_pred = x + f1 * dt + g1 * rand_term
    f2, g2 = get_f_g(t, x_pred, prompts)
    return x + 0.5 * (f1 + f2) * dt + 0.5 * (g1 + g2) * rand_term


@torch.inference_mode()
def ode_step(x, t, prev_t, prompts):
    """Take a single ODE step using a Heun scheme."""
    dt = t - prev_t
    f1, _ = get_f_g(prev_t, x, prompts)
    x_pred = x + f1 * dt
    f2, _ = get_f_g(t, x_pred, prompts)
    return x + 0.5 * (f1 + f2) * dt


def best_stepnum(t, sigma_max=14.6488, sigma_min=2e-3, max_ode_step=18, RHO=7):
    """
    Compute an appropriate number of ODE steps based on the current sigma.
    
    Returns:
        int: The calculated step number.
    """
    A = sigma_min ** (1 / RHO)
    B = sigma_max ** (1 / RHO)
    C = t ** (1 / RHO)
    ratio = (C - A) / (B - A)
    return max(int(np.ceil(max_ode_step * ratio)), 0)


@OdeModeContextManager()
@torch.inference_mode()
def odeint_rest(x, start_t, ts, prompts, max_ode_steps=18):
    """
    Integrate the ODE from start_t using a variable number of steps.
    
    Parameters:
        x: Input latent.
        start_t: The starting time.
        ts: A sequence of timesteps.
        prompts: Guidance conditions.
        max_ode_steps: Maximum steps to take in ODE integration.
    """
    steps = best_stepnum(start_t.item(), max_ode_step=max_ode_steps) + 2
    ts = latent_sde.get_karras_timesteps(steps, start_t, sigma_min=ts[-1])
    prev_t, ts = ts[0], ts[1:]
    for t in ts:
        x = ode_step(x, t, prev_t, prompts)
        prev_t = t
    return x


@OdeModeContextManager()
@torch.inference_mode()
def odeint(x, text_cfg_dict, T, start_t=14.648, end_t=2e-3):
    """
    Integrate the ODE over T steps to update the latent x.
    
    Parameters:
        x: Initial latent.
        text_cfg_dict: Dictionary with 'prompts' and 'cfgs'.
        T: Total number of timesteps.
        start_t: Starting sigma.
        end_t: Minimum sigma.
    """
    ts = latent_sde.get_karras_timesteps(T=T, sigma_max=start_t, sigma_min=end_t)
    cfg_dict = copy.deepcopy(text_cfg_dict)
    cfg_dict['prompts'].append('')
    prompts = {
        'conditions': get_condition(cfg_dict['prompts']),
        'cfgs': cfg_dict['cfgs'],
    }
    prev_t = ts[0]
    for t in ts[1:]:
        x = ode_step(x, t, prev_t, prompts)
        prev_t = t
    return x


@OdeModeContextManager()
@torch.inference_mode()
def oderevert(x, text_cfg_dict, sample_step=100, start_t=14.648, end_t=2e-3):
    """
    Revert integration order by flipping timesteps.
    """
    ts = latent_sde.get_karras_timesteps(T=sample_step, sigma_max=start_t, sigma_min=end_t)
    ts = torch.flip(ts, dims=[0])
    cfg_dict = copy.deepcopy(text_cfg_dict)
    cfg_dict['prompts'].append('')
    prompts = {
        'conditions': get_condition(cfg_dict['prompts']),
        'cfgs': cfg_dict['cfgs'],
    }
    prev_t = ts[0]
    for t in ts[1:]:
        x = ode_step(x, t, prev_t, prompts)
        prev_t = t
    return x


@torch.inference_mode()
def consistency_sampling(x, conds, sample_step=2, start_t=14.648, end_t=2e-3):
    """
    Use the latent_consistency model to update x.
    """
    ts = latent_sde.get_karras_timesteps(T=sample_step, sigma_max=start_t, sigma_min=end_t)
    N = x.shape[0]
    while len(ts) > 1:
        t, ts = ts[0], ts[1:]
        x = latent_consistency(t, x, conds)
        if len(ts) > 1:
            z = torch.randn_like(x[0:1])
            x = x + ts[0] * z.repeat(N, 1, 1, 1)
    return x


@torch.inference_mode()
def sdeint(x, text_cfg_dict, beta, sample_step, start_t=14.648, end_t=2e-3):
    """
    Integrate the SDE using a predictor-corrector scheme.
    """
    latent_sde.change_noise(beta=beta)
    ts = latent_sde.get_karras_timesteps(T=sample_step, sigma_max=start_t, sigma_min=end_t)
    cfg_dict = copy.deepcopy(text_cfg_dict)
    cfg_dict['prompts'].append('')
    prompts = {
        'conditions': get_condition(cfg_dict['prompts']),
        'cfgs': cfg_dict['cfgs'],
    }
    prev_t = ts[0]
    for t in ts[1:]:
        x = sde_step(x, t, prev_t, prompts, torch.randn_like(x))
        prev_t = t
    return x


@torch.inference_mode()
def demon_sampling(x,
                   text_cfg_dict,
                   reward_fn,  # reward function operating on latent space
                   beta,
                   tau,
                   K,
                   T,
                   demon_type="tanh",
                   r_of_c="baseline",
                   c_steps=20,  # Only relevant when r_of_c == "baseline"
                   ode_after=0.11,
                   start_t=14.648,
                   end_t=2e-3,
                   log_dir=None):
    """
    Run the demon sampling process to refine latent x using reward feedback.
    """
    if r_of_c not in ["baseline", "consistency"]:
        raise ValueError(f"Unknown r_of_c: {r_of_c}")
    if x.shape[0] != 1:
        raise ValueError("Input x must have batch size of 1")

    latent_sde.change_noise(beta=beta)
    ts = latent_sde.get_karras_timesteps(T=T, sigma_max=start_t, sigma_min=end_t)

    cfg_dict = copy.deepcopy(text_cfg_dict)
    cfg_dict['prompts'].append('')
    prompts = {
        'conditions': get_condition(cfg_dict['prompts']),
        'cfgs': cfg_dict['cfgs'],
    }

    if r_of_c == "consistency":
        cond = get_condition(cfg_dict['prompts'][:-1])
        if K > 1:
            cond = duplicate_condition(cond, K)

    prev_t, ts = ts[0], ts[1:]
    start_time = datetime.now()

    while len(ts) > 0:
        if prev_t < ode_after:
            x = odeint_rest(x, prev_t, ts, prompts, max_ode_steps=18)
            break

        zs = torch.randn(K, *x.shape[1:], device=x.device, dtype=x.dtype)
        t, ts = ts[0], ts[1:]
        next_xs = sde_step(x, t, prev_t, prompts, zs)

        if r_of_c == "consistency":
            candidates_0 = consistency_sampling(next_xs, cond, sample_step=2, start_t=t)
        elif r_of_c == "baseline":
            candidates_0 = odeint_rest(next_xs, t, ts, prompts, max_ode_steps=c_steps)

        values = torch.tensor(reward_fn(candidates_0),
                              device=x.device,
                              dtype=x.dtype)
        passed_seconds = (datetime.now() - start_time).total_seconds()
        if log_dir is not None:
            with open(f"{log_dir}/expected_reward.txt", "a") as f:
                f.write(f"{values.mean().item()} {values.std().item()} {t.item()} {passed_seconds}\n")

        # Adaptive tau: update if necessary.
        if tau == 'adaptive':
            tau = values.std().item()

        if demon_type == "tanh":
            values = values - values.mean()
            weights = torch.tanh(values / tau)
        elif demon_type == "boltzmann":
            stabilized_values = values - torch.max(values)
            weights = F.softmax(stabilized_values / tau, dim=0)
        elif demon_type == "optimal":
            weights = values
        else:
            raise ValueError(f"Unknown demon_type: {demon_type}")

        # Prevent division by zero or near-zero standard deviation.
        if values.std().item() < 1e-8:
            weights = torch.ones_like(weights)

        z_final = F.normalize(
            (zs * weights.view(-1, 1, 1, 1)).sum(dim=0, keepdim=True),
            dim=(0, 1, 2, 3)
        )
        z_final *= x.numel() ** 0.5
        x = sde_step(x, t, prev_t, prompts, z_final)
        prev_t = t

    return x


def add_noise(latent, t):
    """Add Gaussian noise scaled by t to the latent."""
    return latent + torch.randn_like(latent) * t


def get_init_latent(batch_size=1):
    """Return initial latent states from latent_sde."""
    return latent_sde.prepare_initial_latents(batch_size=batch_size)
