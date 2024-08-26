import os
import shutil

import torch
import torch.distributed as dist
import torch.nn.functional as F
from util.diffusion_coefficients import extract


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)


def q_sample(coeff, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + extract(coeff.sigmas_cum, t, x_start.shape) * noise

    return x_t


def q_sample_pairs(coeff, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = extract(coeff.a_s, t + 1, x_start.shape) * x_t + extract(coeff.sigmas, t + 1, x_start.shape) * noise

    return x_t, x_t_plus_one, noise


def sample_posterior(coefficients, x_0, x_t, t):

    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)

        noise = torch.randn_like(x_t)

        nonzero_mask = 1 - (t == 0).type(torch.float32)

        return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise, noise

    sample_x_pos, noise = p_sample(x_0, x_t, t)

    return sample_x_pos, noise


def sample_from_model(coefficients, generator, n_time, x_init, T, opt):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)

            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(x, t_time, latent_z)
            x_new, noise = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()

    return x


# ------------------------
# Select Phi_star
# ------------------------
def select_phi(name):
    if name == "linear":

        def phi(x):
            return x

    elif name == "kl":

        def phi(x):
            return torch.exp(x)

    elif name == "chi":

        def phi(x):
            y = F.relu(x + 2) - 2
            return 0.25 * y**2 + y

    elif name == "softplus":

        def phi(x):
            return F.softplus(x)

    else:
        raise NotImplementedError

    return phi
