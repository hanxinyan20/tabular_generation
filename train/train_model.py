from copy import deepcopy
import torch
import os
import numpy as np

from diff_scripts.model_tabddpm import GaussianMultinomialDiffusion
from .utils_train import get_model
from .trainer import Trainer

def default_converter(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    elif isinstance(o, (np.floating,)):
        return float(o)
    elif isinstance(o, (np.ndarray,)):
        return o.tolist()
    else:
        return str(o)



def train_model(
    base_dir,
    steps = 10000,
    lr = 8e-5,
    batch_size_per_gpu = 4,
    gradient_accumulate_every = 2, 
    ema_update_every = 10, 
    ema_decay = 0.995,
    adam_betas = (0.9, 0.99),
    weight_decay = 1e-4,
    amp = False,
    mixed_precision_type = 'fp16',
    split_batches = True,
    max_grad_norm = 1.,
    save_and_sample_every = 100,
    num_samples = 9,
    results_folder = './results',
    model_type = 'mlp',
    model_params = None, 
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    seed = 0,
):
    # d_in, num_classes, is_y_cond, rtdl_params, dim_t = 128, padding_to: Optional[int] = None, padding_value: Optional[float] = None
    model = get_model(
        model_type,
        model_params,
    )


    diffusion = GaussianMultinomialDiffusion(
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type, 
        num_timesteps=num_timesteps,
        scheduler=scheduler,
    )


    trainer = Trainer(
        diffusion_model = diffusion,
        base_dir=base_dir,
        batch_size_per_gpu = batch_size_per_gpu,
        gradient_accumulate_every = gradient_accumulate_every,
        train_lr = lr,
        train_num_steps = steps,
        ema_update_every = ema_update_every,
        ema_decay = ema_decay,
        adam_betas = adam_betas,
        save_and_sample_every=save_and_sample_every,
        num_samples=num_samples,
        results_folder = results_folder,
        amp = amp,
        mixed_precision_type = mixed_precision_type,
        split_batches = split_batches,
        max_grad_norm = max_grad_norm,
        seed=seed
    )
    trainer.train()