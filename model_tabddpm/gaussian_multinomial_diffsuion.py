"""
Based on https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
and https://github.com/ehoogeboom/multinomial_diffusion
"""

import torch.nn.functional as F
import torch
import math

import numpy as np
from .utils import *

"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class GaussianMultinomialDiffusion(torch.nn.Module):
    def __init__(
            self,
            denoise_fn,
            num_timesteps=1000,
            gaussian_loss_type='mse',
            gaussian_parametrization='eps',
            multinomial_loss_type='vb_stochastic',
            parametrization='x0',
            scheduler='cosine',
        ):

        super(GaussianMultinomialDiffusion, self).__init__()
        assert multinomial_loss_type in ('vb_stochastic', 'vb_all')
        assert parametrization in ('x0', 'direct')

        if multinomial_loss_type == 'vb_all':
            print('Computing the loss using the bound on _all_ timesteps.'
                  ' This is expensive both in terms of memory and computation.')

        # num_numerical_features = num_numerical_features
        # num_classes = num_classes # it as a vector [K1, K2, ..., Km]
        # num_classes_expanded = torch.from_numpy(
        #     np.concatenate([num_classes[i].repeat(num_classes[i]) for i in range(len(num_classes))])
        # ).to(device)

        # self.slices_for_classes = [np.arange(num_classes[0])]
        # offsets = np.cumsum(num_classes)
        # for i in range(1, len(offsets)):
        #     self.slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
        # self.offsets = torch.from_numpy(np.append([0], offsets)).to(device)

        self._denoise_fn = denoise_fn
        self.gaussian_loss_type = gaussian_loss_type
        self.gaussian_parametrization = gaussian_parametrization
        self.multinomial_loss_type = multinomial_loss_type
        self.num_timesteps = num_timesteps
        self.parametrization = parametrization
        self.scheduler = scheduler

        alphas = 1. - get_named_beta_schedule(scheduler, num_timesteps)
        alphas = torch.tensor(alphas.astype('float64'))
        betas = 1. - alphas

        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.tensor(np.append(1.0, alphas_cumprod[:-1]))
        alphas_cumprod_next = torch.tensor(np.append(alphas_cumprod[1:], 0.0))
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)

        # Gaussian diffusion

        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.from_numpy(
            np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        ).float()
        self.posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).float()
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev)
            * np.sqrt(alphas.numpy())
            / (1.0 - alphas_cumprod)
        ).float()

        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        # Convert to float32 and register buffers.
        self.register_buffer('alphas', alphas.float())
        self.register_buffer('log_alpha', log_alpha.float())
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float())
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float())
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float())
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float())
        self.register_buffer('alphas_cumprod_next', alphas_cumprod_next.float())
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod.float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod.float())
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod.float())
        self.register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod.float())

        self.register_buffer('Lt_history', torch.zeros(num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(num_timesteps))
    
    # Gaussian part
    def gaussian_q_mean_variance(self, x_start, t):
        mean = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_1_min_cumprod_alpha, t, x_start.shape
        )
        return mean, variance, log_variance
    
    def gaussian_q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
    
    def gaussian_q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    @property
    def device(self):
        return self.betas.device
    def gaussian_p_mean_variance(
        self, model_output, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None
    ):
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        model_variance = torch.cat([self.posterior_variance[1].unsqueeze(0).to(x.device), (1. - self.alphas)[1:]], dim=0)
        # model_variance = self.posterior_variance.to(x.device)
        model_log_variance = torch.log(model_variance)

        model_variance = extract(model_variance, t, x.shape)
        model_log_variance = extract(model_log_variance, t, x.shape)


        # model_variance_all     = self.posterior_variance.to(x.device)      # [T]
        # model_log_variance_all = torch.log(model_variance_all)

        # model_variance     = extract(model_variance_all,     t, x.shape)
        # model_log_variance = extract(model_log_variance_all, t, x.shape)



        if self.gaussian_parametrization == 'eps':
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        elif self.gaussian_parametrization == 'x0':
            pred_xstart = model_output
        else:
            raise NotImplementedError
        model_mean, _, _ = self.gaussian_q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        ), f'{model_mean.shape}, {model_log_variance.shape}, {pred_xstart.shape}, {x.shape}'

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
    
    def _vb_terms_bpd(
        self, model_output, x_start, x_t, t, clip_denoised=False, model_kwargs=None
    ):
        true_mean, _, true_log_variance_clipped = self.gaussian_q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.gaussian_p_mean_variance(
            model_output, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"], "out_mean": out["mean"], "true_mean": true_mean}
    
    def _prior_gaussian(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.gaussian_q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)
    
    def _gaussian_loss(self, model_out, x_start, x_t, t, noise, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        terms = {}
        if self.gaussian_loss_type == 'mse':
            terms["loss"] = mean_flat((noise - model_out) ** 2)
        elif self.gaussian_loss_type == 'kl':
            terms["loss"] = self._vb_terms_bpd(
                model_output=model_out,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]


        return terms['loss']
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def gaussian_p_sample(
        self,
        model_out,
        x,
        t,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
    ):
        out = self.gaussian_p_mean_variance(
            model_out,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        ) # out should be 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-\hat{alpha_t}) * eps , where eps is predicted noise
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise # exp(0.5 * out["log_variance"]) is \sigma_t
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    # Multinomial part

    def multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t, num_classes_expanded):
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)
        num_classes_expanded = torch.tensor(num_classes_expanded, device=log_x_t.device)
        num_classes_expanded = num_classes_expanded.unsqueeze(0).repeat(log_x_t.shape[0], 1)
        # alpha_t * E[xt] + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - torch.log(num_classes_expanded)
        )

        return log_probs

    def q_pred(self, log_x_start, t, num_classes_expanded):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)
        num_classes_expanded = torch.tensor(num_classes_expanded, device=log_x_start.device)
        num_classes_expanded = num_classes_expanded.unsqueeze(0).repeat(log_x_start.shape[0], 1)
        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - torch.log(num_classes_expanded)
        )

        return log_probs

    def predict_start(self, model_out, log_x_t, num_classes):


        assert model_out.size(0) == log_x_t.size(0)
        assert model_out.size(1) == sum(num_classes) , f'{model_out.size()}'

        log_pred = torch.empty_like(model_out)
        slices_for_classes = [np.arange(num_classes[0])]
        offsets = np.cumsum(num_classes)
        for i in range(1, len(offsets)):
            slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
        for ix in slices_for_classes:
            logits = model_out[:, ix]
            log_probs = F.log_softmax(logits.float(), dim=1)   # 计算用 fp32 更稳
            log_pred[:, ix] = log_probs.to(logits.dtype)   
            # log_pred[:, ix] = F.log_softmax(model_out[:, ix], dim=1)
        return log_pred

    def q_posterior(self, log_x_start, log_x_t, t, num_classes, num_classes_expanded):
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1, num_classes_expanded)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.to(log_x_start.device).view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0.to(torch.float32))

        # unnormed_logprobs = log_EV_qxtmin_x0 +
        #                     log q_pred_one_timestep(x_t, t)
        # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
        # Not very easy to see why this is true. But it is :)
        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t, num_classes_expanded)
        offsets = np.cumsum(num_classes)
        offsets = torch.from_numpy(np.append([0], offsets)).to(log_x_start.device)
        log_EV_xtmin_given_xt_given_xstart = \
            unnormed_logprobs \
            - sliced_logsumexp(unnormed_logprobs, offsets)

        return log_EV_xtmin_given_xt_given_xstart

    def p_pred(self, model_out, log_x, t, num_classes, num_classes_expanded):
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(model_out, log_x, num_classes=num_classes)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t, num_classes=num_classes, num_classes_expanded=num_classes_expanded)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(model_out, log_x, num_classes=num_classes)
        else:
            raise ValueError
        return log_model_pred

    @torch.no_grad()
    def p_sample(self, model_out, log_x, t, num_classes, num_classes_expanded):
        model_log_prob = self.p_pred(model_out, log_x=log_x, t=t, num_classes=num_classes, num_classes_expanded=num_classes_expanded)
        out = self.log_sample_categorical(model_log_prob, num_classes=num_classes)
        return out

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = self.log_alpha.device

        b = shape[0]
        # start with random normal image.
        img = torch.randn(shape, device=device)

        for i in reversed(range(1, self.num_timesteps)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    @torch.no_grad()
    def _sample(self, image_size, batch_size = 16):
        return self.p_sample_loop((batch_size, 3, image_size, image_size))

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in reversed(range(0, t)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def log_sample_categorical(self, logits, num_classes):
        full_sample = []
        slices_for_classes = [np.arange(num_classes[0])]
        offsets = np.cumsum(num_classes)
        for i in range(1, len(offsets)):
            slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
        
        for i in range(len(num_classes)):
            one_class_logits = logits[:, slices_for_classes[i]]
            uniform = torch.rand_like(one_class_logits)
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
            sample = (gumbel_noise + one_class_logits).argmax(dim=1)
            full_sample.append(sample.unsqueeze(1))
        full_sample = torch.cat(full_sample, dim=1)
        log_sample = index_to_log_onehot(full_sample, num_classes)
        return log_sample

    def q_sample(self, log_x_start, t, num_classes_expanded, num_classes):
        log_EV_qxt_x0 = self.q_pred(log_x_start, t, num_classes_expanded)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0, num_classes)

        return log_sample


    def kl_prior(self, log_x_start, num_classes_expanded):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones, num_classes_expanded=num_classes_expanded)
        num_classes_expanded = torch.tensor(num_classes_expanded, device=log_x_start.device)
        num_classes_expanded = num_classes_expanded.unsqueeze(0).repeat(log_x_start.shape[0], 1)
        log_half_prob = -torch.log(num_classes_expanded * torch.ones_like(log_qxT_prob))

        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)
        return sum_except_batch(kl_prior)

    def compute_Lt(self, model_out, log_x_start, log_x_t, t, num_classes, num_classes_expanded, detach_mean=False):
        log_true_prob = self.q_posterior(
            log_x_start=log_x_start, log_x_t=log_x_t, t=t, num_classes=num_classes, num_classes_expanded=num_classes_expanded)
        log_model_prob = self.p_pred(model_out, log_x=log_x_t, t=t, num_classes=num_classes, num_classes_expanded=num_classes_expanded)

        if detach_mean:
            log_model_prob = log_model_prob.detach()

        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1. - mask) * kl

        return loss

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = (Lt_sqrt / Lt_sqrt.sum()).to(device)

            t = torch.multinomial(pt_all, num_samples=b, replacement=True).to(device)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def _multinomial_loss(self, model_out, log_x_start, log_x_t, t, pt, num_classes_expanded, num_classes):

        if self.multinomial_loss_type == 'vb_stochastic':
            kl = self.compute_Lt(
                model_out, log_x_start, log_x_t, t, num_classes, num_classes_expanded
            )
            kl_prior = self.kl_prior(log_x_start, num_classes_expanded)
            vb_loss = kl / pt + kl_prior
            return vb_loss
        else:
            raise ValueError()
    
    def mixed_loss(self, batch:dict):
        X, y, emb, meta = batch['X'], batch.get('y', None), batch.get('emb', None), batch.get('meta', None)
        print("X[:10, 0]:", X[:10, 0])
        num_numerical_features = meta['num_numerical_features']
        num_categories_expanded = meta['num_categories_expanded']
        num_categories = meta['num_categories']
        b = X.shape[0]
        device = X.device
        t, pt = self.sample_time(b, device, 'uniform')

        x_num = X[:, :num_numerical_features]
        x_cat = X[:, num_numerical_features:]
        
        x_num_t = x_num
        log_x_cat_t = x_cat
        if x_num.shape[1] > 0:
            noise = torch.randn_like(x_num)
            x_num_t = self.gaussian_q_sample(x_num, t, noise=noise) # 给 x_0 加噪后在t时刻的值 
        if x_cat.shape[1] > 0:
            # log_x_cat = index_to_log_onehot(x_cat.long(), num_classes)
            log_x_cat = torch.log(x_cat.float().clamp(min=1e-30)) # log one-hot
            log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t, num_classes_expanded=num_categories_expanded, num_classes=num_categories)
        
        x_in = torch.cat([x_num_t, log_x_cat_t], dim=1)
        model_out = self._denoise_fn(
            x_in,
            t,
            y,
            emb,
        )
        model_out_num = model_out[:, :num_numerical_features]
        print("model_out_num[:10, 0]:", model_out_num[:10, 0])
        model_out_cat = model_out[:, num_numerical_features:]
        # 检查是否有nan
        if torch.isnan(model_out_num).any():
            print('model_out_num has nan')
            print('x_num', x_num)
            print('x_num_t', x_num_t)
        if torch.isnan(model_out_cat).any():
            print('model_out_cat has nan')
            print('x_cat', x_cat)
            print('log_x_cat', log_x_cat)
            print('log_x_cat_t', log_x_cat_t)

        loss_multi = torch.zeros((1,)).float()
        loss_gauss = torch.zeros((1,)).float()
        if x_cat.shape[1] > 0:
            loss_multi = self._multinomial_loss(model_out_cat, log_x_cat, log_x_cat_t, t, pt, num_categories_expanded, num_categories) / len(num_categories)

        if x_num.shape[1] > 0:
            loss_gauss = self._gaussian_loss(model_out_num, x_num, x_num_t, t, noise)

        return loss_multi.mean(), loss_gauss.mean()

    @torch.no_grad()
    def sample(self, num_samples, y, emb, num_classes_expanded, num_numerical_features, num_classes):
        b = num_samples
        device = self.log_alpha.device
        z_norm = torch.randn((b, num_numerical_features), device=device)

        has_cat = num_classes[0] != 0
        log_z = torch.zeros((b, 0), device=device).float()
        if has_cat:
            uniform_logits = torch.zeros((b, len(num_classes_expanded)), device=device)
            log_z = self.log_sample_categorical(uniform_logits, num_classes=num_classes)
        print("log_z:", log_z[0])
        
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Sample timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = self._denoise_fn(
                torch.cat([z_norm, log_z], dim=1).float(),
                t,
                y,
                emb
            ).to(device)
            model_out_num = model_out[:, :num_numerical_features]
            model_out_cat = model_out[:, num_numerical_features:]
            print(f"timestep {i} model_out_num[:10, 0]:{model_out_num[:10, 0]}")
            z_norm = self.gaussian_p_sample(model_out_num, z_norm, t, clip_denoised=False)['sample']
            if has_cat:
                log_z = self.p_sample(model_out_cat, log_z, t, num_classes, num_classes_expanded)

        print()
        z_ohe = torch.exp(log_z).round()

        sample = torch.cat([z_norm, z_ohe], dim=1).cpu()
        return sample
    
    # def sample_all(self, num_samples, batch_size, y_dist, ddim=False):
    #     if ddim:
    #         print('Sample using DDIM.')
    #         sample_fn = self.sample_ddim
    #     else:
    #         sample_fn = self.sample
        
    #     b = batch_size

    #     all_y = []
    #     all_samples = []
    #     num_generated = 0
    #     while num_generated < num_samples:
    #         sample, out_dict = sample_fn(b, y_dist)
    #         mask_nan = torch.any(sample.isnan(), dim=1)
    #         sample = sample[~mask_nan]
    #         out_dict['y'] = out_dict['y'][~mask_nan]

    #         all_samples.append(sample)
    #         all_y.append(out_dict['y'].cpu())
    #         if sample.shape[0] != b:
    #             raise FoundNANsError
    #         num_generated += sample.shape[0]

    #     x_gen = torch.cat(all_samples, dim=0)[:num_samples]
    #     y_gen = torch.cat(all_y, dim=0)[:num_samples]

    #     return x_gen, y_gen