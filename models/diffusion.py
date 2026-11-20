import torch
from torch import nn
from torch.nn import functional as F

from configs.model import ModelConfig
from models.initialization import STRATEGIES, channel_guidance, initialize_mask


def structure_aware_initialization(target_image: torch.Tensor, coarse_mask: torch.Tensor):
    """Equation in Sec. Structure-aware Learning: G * coarse + G * channel prior."""
    return initialize_mask(coarse_mask, channel_guidance(target_image, coarse_mask), "joint")


class GaussianDiffusion(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        betas = torch.linspace(config.beta_start, config.beta_end, config.diffusion_steps)
        self.register_buffer("alpha_bar", torch.cumprod(1 - betas, dim=0))

    def q_sample(self, clean_mask: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        alpha = self.alpha_bar[t].view(-1, 1, 1, 1)
        return alpha.sqrt() * clean_mask + (1 - alpha).sqrt() * noise

    def training_loss(self, model, clean_mask: torch.Tensor, image: torch.Tensor, use_pff: bool = True,
                      structural_cues=None):
        clean_mask = clean_mask * 2 - 1
        t = torch.randint(0, self.config.diffusion_steps, (clean_mask.shape[0],), device=clean_mask.device)
        noise = torch.randn_like(clean_mask)
        noisy_mask = self.q_sample(clean_mask, t, noise)
        condition_features = (model.encode_condition(image) if structural_cues is None else
                              model.encode_condition(image, structural_cues=structural_cues))
        prediction = model(noisy_mask, t, condition_features, use_pff=use_pff)
        return F.mse_loss(prediction, noise)

    @torch.no_grad()
    def ddim_sample(self, model, target_image: torch.Tensor, coarse_mask: torch.Tensor,
                    condition_image: torch.Tensor, steps: int = 50, eta: float = 0.0,
                    use_sal: bool = True, use_pff: bool = True, initialization=None,
                    return_features: bool = False):
        if not 10 <= steps <= 500:
            raise ValueError("The manuscript specifies 10-500 DDIM sampling steps")
        if not 0 <= eta <= 1:
            raise ValueError("DDIM eta must be in [0, 1]")
        strategy = initialization or ("joint" if use_sal else "gaussian")
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown initialization: {strategy}")
        extra_conditions = getattr(model, "cue_conditioner", None) is not None
        if extra_conditions != (strategy == "additional_condition"):
            raise ValueError("additional_condition requires its own trained cue-conditioning checkpoint")
        prior = channel_guidance(target_image, coarse_mask)
        condition_features = (model.encode_condition(condition_image, structural_cues=(coarse_mask, prior))
                              if extra_conditions else model.encode_condition(condition_image))
        sample = initialize_mask(coarse_mask, prior, strategy)
        last_features = None
        timesteps = torch.linspace(self.config.diffusion_steps - 1, 0, steps,
                                   device=sample.device).long().tolist()
        for index, time in enumerate(timesteps):
            previous = timesteps[index + 1] if index + 1 < len(timesteps) else -1
            t = torch.full((sample.shape[0],), time, device=sample.device, dtype=torch.long)
            if return_features and previous < 0:
                epsilon, last_features = model(sample, t, condition_features, use_pff=use_pff,
                                                return_features=True)
            else:
                epsilon = model(sample, t, condition_features, use_pff=use_pff)
            alpha = self.alpha_bar[time]
            alpha_previous = self.alpha_bar[previous] if previous >= 0 else torch.ones_like(alpha)
            clean = ((sample - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()).clamp(-1, 1)
            sigma = eta * ((1 - alpha_previous) / (1 - alpha) *
                           (1 - alpha / alpha_previous)).clamp_min(0).sqrt()
            direction = (1 - alpha_previous - sigma.square()).clamp_min(0).sqrt() * epsilon
            sample = alpha_previous.sqrt() * clean + direction
            if previous >= 0 and eta > 0:
                sample = sample + sigma * torch.randn_like(sample)
        probabilities = ((sample + 1) / 2).clamp(0, 1)
        return (probabilities, last_features) if return_features else probabilities
