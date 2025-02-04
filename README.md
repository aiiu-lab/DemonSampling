# Sampling Demon

**Official implementation of "Sampling Demon" (arXiv:2410.05760).**

This repository contains the official implementation of Sampling Demon, an inference-time, backpropagation-free preference alignment method for diffusion models. By aligning the denoising process with user preferences via stochastic optimization, Sampling Demon enables the use of non-differentiable reward signals—such as those from Visual-Language Model (VLM) APIs and human judgements—without requiring retraining or fine-tuning of the underlying diffusion model.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Running Provided Pipelines](#running-provided-pipelines)
  - [LLM and Manual Selection Pipelines](#llm-and-manual-selection-pipelines)
  - [Custom Implementations](#custom-implementations)
- [Low-Level API](#low-level-api)
- [Experiments and Results](#experiments-and-results)
- [Credits and Acknowledgments](#credits-and-acknowledgments)
- [Citation](#citation)
- [License](#license)

---

## Overview

Diffusion models have revolutionized image generation; however, aligning these models with diverse user preferences remains a significant challenge. Traditional approaches rely either on costly retraining or require differentiable reward functions, limiting their scope when using non-differentiable sources such as VLM APIs and human feedback.

**Sampling Demon** overcomes these limitations by steering the denoising process via stochastic optimization at inference time. Inspired by Maxwell's Demon, our method evaluates multiple candidate noise perturbations and selectively synthesizes the ones that yield higher rewards. Our contributions highlight:

- **Backpropagation-Free Alignment:** Incorporate non-differentiable reward signals directly into the inference process.
- **Plug-and-Play Integration:** Seamlessly integrate with existing diffusion models without additional training.
- **Theoretical and Empirical Validation:** We provide both theoretical insights and comprehensive experimental evidence showing significant improvements in aesthetic scores.
- **Broad Applicability:** Our approach supports reward signals from various sources, including VLM APIs and human judgements.

---

## Installation

To install the required packages, run:

```bash
conda env create -f environment.yml  # The build takes about 30 minutes on our machine :(
pip install -e .
```

> **Note:**  
> If you experience issues with PyTorch versioning, try uninstalling torch-related packages and reinstall using:
>
> ```bash
> pip3 install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1
> ```
>
> Alternatively, install the packages from `requirements.txt` one by one if needed.

---

## Usage

### Running Provided Pipelines

The repository includes several example pipelines that demonstrate Sampling Demon in action. These pipelines illustrate how to align diffusion models with user preferences using various reward functions.

#### Running VLM as Demon Pipeline

This pipeline leverages a Visual-Language Model (VLM) as the reward function to generate images:

```bash
python pipelines/vllm_generate.py --model "gemini" --K 16 --T 128 --beta 0.1
```

#### Running Manual Selection Pipeline

Interact with the algorithm via the manual selection pipeline, which provides a user interface for selecting preferred outcomes:

```bash
python pipelines/choose_generate.py --text "A boulder in elevator" --K 16 --T 128
```

### Custom Implementations

Develop your own pipeline by subclassing the `DemonGenerater` abstract class. Override the `rewards` method to integrate your custom reward function (i.e., mapping a list of PIL images to reward scores). See the examples in `pipelines/vllm_generate.py` and `pipelines/choose_generate.py` for guidance.

---

## Low-Level API

For advanced users who wish to modify Sampling Demon at a lower level, we provide a low-level API that was integral to our research. The following snippets demonstrate key functionalities:

### ODE Integral

```python
condition = {
    "prompts": ["On Moon", "Astronaut", "Riding a donkey"],
    "cfgs": [3, 2, 4]
}
steps = 20
x = get_init_latent()  # sigma is 14.6488 for Stable Diffusion
x = odeint(x, condition, steps)
pil = from_latent_to_pil(x)
```

### ODE Reverse

```python
condition = {
    "prompts": ["An astronaut riding a horse on Mars."],
    "cfgs": [5]
}
x = from_pil_to_latent(pil)
x = oderevert(x, condition)
x = odeint(x, condition, 20)
pil = from_latent_to_pil(x)
```

### SDEdit

```python
old_condition = {
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
x = from_pil_to_latent(pil)
x = oderevert(x, old_condition, start_t=sigma)
x = sdeint(x, new_condition, beta, steps, start_t=sigma)
pil = from_latent_to_pil(x)
```

---

## Miscellaneous
1. **For SDv1.5:**
    - Please switch to the `mini` branch for the SDv1.5-compatible version of the code.
    - `pipelines/` is compatible with SDv1.5.
2. **Running the test:**
    - pytest is used for testing. To run the tests, use the command `pytest tests`
    - Specifically, the low-level API demonstration is identical to `tests/test_api.py`.

---

## Credits and Acknowledgments

- **Aesthetic Model Checkpoint:** Provided by [DDPO](https://github.com/kvablack/ddpo-pytorch/tree/main).
- **Safety Checker:** Utilizes the Stable Diffusion Safety Checker from CompVis.
- **Contributors:**  
  For questions or suggestions, please raise an issue or contact the [author](mailto:rareone0602@gmail.com).

---

## Citation

If you find this code useful in your research, please consider citing our paper:

```
@inproceedings{
yeh2025trainingfree,
title={Training-Free Diffusion Model Alignment with Sampling Demons},
author={Po-Hung Yeh, Kuang-Huei Lee, Jun-cheng Chen},
booktitle={International Conference on Learning Representations},
year={2025},
}
```

---

## License

This project is licensed under the Apache License 2.0 license. 
