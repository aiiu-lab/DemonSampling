import torch
import torch.nn as nn

from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from transformers import CLIPImageProcessor


safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="safety_checker",
                use_safetensors=True,
                variant="fp16",
            ).to('cuda')
feature_extractor = CLIPImageProcessor.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="feature_extractor",
        )

def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())

def safety_check_batch(images):
    clip_input = feature_extractor(images, return_tensors="pt").pixel_values.to('cuda')
    pooled_output = safety_checker.vision_model(clip_input)[1]  # pooled_output
    image_embeds = safety_checker.visual_projection(pooled_output)

    special_cos_dist = cosine_distance(image_embeds, safety_checker.special_care_embeds)
    cos_dist = cosine_distance(image_embeds, safety_checker.concept_embeds)
    _, has_nsfw = safety_checker(
            images=clip_input,  # dummy input
            clip_input=clip_input,
        )
    return has_nsfw

if __name__ == "__main__":
    # Create 8 black images and check if they are safe
    from PIL import Image
    black_image = Image.open('out.png')
    images = [black_image] * 8
    safety_check_batch(images)