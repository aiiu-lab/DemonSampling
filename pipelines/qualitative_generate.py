# Third-party library imports
import fire
import torch
import json
from transformers import AutoModel, AutoProcessor

# Local application/library specific imports
import importlib
from generate_abstract import DemonGenerater
import hpsv2

# Global prompt variable to be set in qualitative_generate.
prompt = None

# Lazy model getters.
_aesthetic_scorer = None
def get_aesthetic_scorer():
    global _aesthetic_scorer
    if _aesthetic_scorer is None:
        from reward_models.AestheticScorer import AestheticScorer
        _aesthetic_scorer = AestheticScorer().to('cuda')
    return _aesthetic_scorer

_pickscore_processor = None
_pickscore_model = None
def get_pickscore_models():
    global _pickscore_processor, _pickscore_model
    if _pickscore_processor is None or _pickscore_model is None:
        _pickscore_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        _pickscore_model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to('cuda')
    return _pickscore_processor, _pickscore_model

_imageReward_model = None
def get_imageReward_model():
    global _imageReward_model
    if _imageReward_model is None:
        # Import ImageReward lazily.
        RM = importlib.import_module("ImageReward")
        _imageReward_model = RM.load("ImageReward-v1.0")
    return _imageReward_model

@torch.inference_mode()
def hpsv2_reward(pil):
    """Compute reward using hpsv2 and scale it appropriately."""
    return hpsv2.score(pil, prompt, hps_version="v2.1")[0] * 40

@torch.inference_mode()
def rm_reward(pil):
    """Compute reward using ImageReward."""
    model = get_imageReward_model()
    return model.score(prompt, [pil])

@torch.inference_mode()
def pickscore_reward(pil):
    """Compute reward using PickScore."""
    processor, model = get_pickscore_models()
    inputs = processor(images=pil, text=prompt, return_tensors="pt", padding=True).to('cuda')
    return model(**inputs).logits_per_image.item()

@torch.inference_mode()
def aesthetic_reward(pil):
    """Compute aesthetic reward."""
    scorer = get_aesthetic_scorer()
    return scorer(pil).item()

def qualitative_generate(
    beta=0.1,
    tau='adaptive',
    K=16,
    T=64,
    demon_type="tanh",
    r_of_c="baseline",
    c_steps=20,
    ode_after=0.11,
    text="U uwawa uwa",
    cfg=2,
    seed=None,
    save_pils=False,
    aesthetic=False,
    imagereward=False,
    pickscore=False,
    hpsv2_flag=False,
    experiment_directory="experiments/qualitative_generate",
):
    """
    Generate images qualitatively using demon sampling and evaluate them via one or more reward models.
    
    Args:
        beta, tau, K, T, demon_type, r_of_c, c_steps, ode_after, cfg, seed, save_pils: 
            Parameters controlling the sampling process.
        text (str): Text prompt.
        aesthetic, imagereward, pickscore, hpsv2_flag (bool): Flags for which reward functions to apply.
        experiment_directory (str): Base directory for outputs.
    """
    global prompt
    prompt = text

    def reward(pil):
        total = 0
        if aesthetic:
            total += aesthetic_reward(pil)
        if imagereward:
            total += rm_reward(pil)
        if pickscore:
            total += pickscore_reward(pil)
        if hpsv2_flag:
            total += hpsv2_reward(pil)
        return total

    class QualitativeGenerater(DemonGenerater):
        def rewards(self, pils):
            """Compute rewards for each generated PIL image."""
            return [reward(pil) for pil in pils]
        
        def generate(self, prompt):
            """
            Override generate to update the config file with reward flags.
            Uses ODE-only sampling if no reward model is active.
            """
            super().generate(prompt, ode=not any([aesthetic, imagereward, pickscore, hpsv2_flag]))
            # Update config file with reward options.
            config_path = f'{self.log_dir}/config.json'
            with open(config_path, 'r') as f:
                config = json.load(f)
            config.update({
                'aesthetic': aesthetic,
                'imagereward': imagereward,
                'pickscore': pickscore,
                'hpsv2': hpsv2_flag
            })
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)

    # Instantiate the qualitative generator using updated parameter names.
    generator = QualitativeGenerater(
        cfg=cfg,
        beta=beta,
        tau=tau,
        K=K,
        T=T,
        demon_type=demon_type,
        r_of_c=r_of_c,
        c_steps=c_steps,
        ode_after=ode_after,
        seed=seed,
        save_pils=save_pils,
        experiment_directory=experiment_directory
    )

    generator.generate(prompt=text)

if __name__ == '__main__':
    fire.Fire(qualitative_generate)

# Run the following command to generate images:
# python pipelines/qualitative_generate.py --text "An astronaut riding a horse on Mars." --aesthetic --imagereward --pickscore --hpsv2_flag
