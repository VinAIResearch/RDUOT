# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import os
import shutil

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.multiprocessing import Process
from tqdm import tqdm
from util.args_parser import args_parser
from util.data_process import getCleanData, getMixedData
from util.diffusion_coefficients import get_sigma_schedule, get_time_schedule
from util.utility import broadcast_params, copy_source, q_sample_pairs, sample_from_model, sample_posterior, select_phi


class Diffusion_Coefficients:
    def __init__(self, args, device):

        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum**2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1

        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)


# %% posterior sampling
class Posterior_Coefficients:
    def __init__(self, args, device):

        _, _, self.betas = get_sigma_schedule(args, device=device)

        # we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (
                torch.tensor([1.0], dtype=torch.float32, device=device),
                self.alphas_cumprod[:-1],
            ),
            0,
        )
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (
            (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod)
        )

        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))


# %%
def train(rank, gpu, args):
    from EMA import EMA
    from score_sde.models.discriminator import Discriminator_64, Discriminator_large, Discriminator_small
    from score_sde.models.ncsnpp_generator_adagn import NCSNpp

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device("cuda:{}".format(gpu))

    batch_size = args.batch_size

    nz = args.nz  # latent dimension

    if args.perturb_dataset == "none":
        dataset = getCleanData(args.dataset, image_size=args.image_size)
    else:
        dataset = getMixedData(
            args.dataset,
            args.perturb_dataset,
            percentage=args.perturb_percent,
            image_size=args.image_size,
            shuffle=args.shuffle,
        )

    print("Finish loading dataset")

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size, rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    netG = NCSNpp(args).to(device)

    if args.dataset in ["cifar10", "mnist", "stackmnist", "stl10", "celeba_64"]:
        print("using small discriminator")
        netD = Discriminator_small(
            nc=2 * args.num_channels,
            ngf=args.ngf,
            t_emb_dim=args.t_emb_dim,
            act=nn.LeakyReLU(0.2),
        ).to(device)
    elif args.dataset in ["clipart", "quickdraw", "sketch"]:
        print("using 64 discriminator")
        netD = Discriminator_64(
            nc=2 * args.num_channels,
            ngf=args.ngf,
            t_emb_dim=args.t_emb_dim,
            act=nn.LeakyReLU(0.2),
        ).to(device)
    else:
        print("using large discriminator")
        netD = Discriminator_large(
            nc=2 * args.num_channels,
            ngf=args.ngf,
            t_emb_dim=args.t_emb_dim,
            act=nn.LeakyReLU(0.2),
        ).to(device)

    broadcast_params(netG.parameters())
    broadcast_params(netD.parameters())

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))

    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)

    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.schedule, eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.schedule, eta_min=1e-5)

    # ddp
    netG = nn.parallel.DistributedDataParallel(netG, device_ids=[gpu])
    netD = nn.parallel.DistributedDataParallel(netD, device_ids=[gpu])

    exp = args.exp

    algo = "rduot"
    parent_dir = f"./saved_info/{algo}/{args.dataset}"

    if args.perturb_percent > 0:
        parent_dir += f"_{int(args.perturb_percent)}p_{args.perturb_dataset}"

    parent_dir += f"/{args.version}"

    if exp == "none":
        exp_path = parent_dir
    else:
        exp_path = os.path.join(parent_dir, exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)
            shutil.copytree("score_sde/models", os.path.join(exp_path, "score_sde/models"))

    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)

    if args.resume:
        checkpoint_file = os.path.join(exp_path, "content.pth")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint["epoch"]
        epoch = init_epoch
        netG.load_state_dict(checkpoint["netG_dict"])
        # load G

        optimizerG.load_state_dict(checkpoint["optimizerG"])
        schedulerG.load_state_dict(checkpoint["schedulerG"])
        # load D
        netD.load_state_dict(checkpoint["netD_dict"])
        optimizerD.load_state_dict(checkpoint["optimizerD"])
        schedulerD.load_state_dict(checkpoint["schedulerD"])
        global_step = checkpoint["global_step"]
        print("=> loaded checkpoint (epoch {})".format(checkpoint["epoch"]))
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    # get phi star
    phi_star1 = select_phi(args.phi1)
    phi_star2 = select_phi(args.phi2)

    for epoch in range(init_epoch, args.num_epoch + 1):
        train_sampler.set_epoch(epoch)

        for iteration, (x, y) in enumerate(tqdm(data_loader)):
            for p in netD.parameters():
                p.requires_grad = True

            netD.zero_grad()

            # sample from p(x_0)
            real_data = x.to(device, non_blocking=True)

            # sample t
            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)

            x_t, x_tp1, noise = q_sample_pairs(coeff, real_data, t)
            x_t.requires_grad = True

            # train with real
            D_real = netD(x_t, t, x_tp1.detach())

            errD_real = phi_star2(-D_real)
            errD_real = errD_real.mean()
            errD_real.backward(retain_graph=True)

            if args.lazy_reg is None:
                grad_real = torch.autograd.grad(outputs=D_real.sum(), inputs=x_t, create_graph=True)[0]
                grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()

                grad_penalty = args.r1_gamma / 2 * grad_penalty
                grad_penalty.backward()
            else:
                if global_step % args.lazy_reg == 0:

                    grad_real = torch.autograd.grad(outputs=D_real.sum(), inputs=x_t, create_graph=True)[0]
                    grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad_penalty = args.r1_gamma / 2 * grad_penalty
                    grad_penalty.backward()

            # train with fake
            latent_z = torch.randn(batch_size, nz, device=device)

            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample, _ = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

            output = netD(x_pos_sample, t, x_tp1.detach())

            errD_fake = phi_star1(
                output
                - args.tau
                * torch.sum(
                    ((x_0_predict - x_tp1.detach()).view(x_tp1.detach().size(0), -1)) ** 2,
                    dim=1,
                )
            )
            errD_fake = errD_fake.mean()
            errD_fake.backward()

            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            # update G
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()

            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)

            x_t, x_tp1, _ = q_sample_pairs(coeff, real_data, t)

            latent_z = torch.randn(batch_size, nz, device=device)

            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample, noise = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

            output = netD(x_pos_sample, t, x_tp1.detach())

            errG = (
                args.tau
                * torch.sum(
                    ((x_0_predict - x_tp1.detach()).view(x_tp1.detach().size(0), -1)) ** 2,
                    dim=1,
                )
                - output
            )
            errG = errG.mean()

            errG.backward()
            optimizerG.step()

            global_step += 1
            if iteration % 100 == 0:
                if rank == 0:
                    print(
                        "epoch {} iteration{}, G Loss: {}, D Loss: {}".format(
                            epoch, iteration, errG.item(), errD.item()
                        )
                    )

        if not args.no_lr_decay:

            schedulerG.step()
            schedulerD.step()

        if rank == 0:
            if epoch % 10 == 0:
                torchvision.utils.save_image(
                    x_pos_sample,
                    os.path.join(exp_path, "xpos_epoch_{}.png".format(epoch)),
                    normalize=True,
                )

            x_t_1 = torch.randn_like(real_data)
            fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, T, args)
            if epoch % 10 == 0:
                torchvision.utils.save_image(
                    fake_sample,
                    os.path.join(exp_path, "sample_discrete_epoch_{}.png".format(epoch)),
                    normalize=True,
                )

            if args.save_content:
                if epoch % args.save_content_every == 0:
                    print("Saving content.")
                    content = {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "args": args,
                        "netG_dict": netG.state_dict(),
                        "optimizerG": optimizerG.state_dict(),
                        "schedulerG": schedulerG.state_dict(),
                        "netD_dict": netD.state_dict(),
                        "optimizerD": optimizerD.state_dict(),
                        "schedulerD": schedulerD.state_dict(),
                    }

                    torch.save(content, os.path.join(exp_path, "content.pth"))

            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)

                torch.save(
                    netG.state_dict(),
                    os.path.join(exp_path, "netG_{}.pth".format(epoch)),
                )
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)


def init_processes(rank, size, fn, args):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = args.master_address
    os.environ["MASTER_PORT"] = args.master_port
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()


def cleanup():
    dist.destroy_process_group()


# %%
if __name__ == "__main__":
    args = args_parser()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print("Node rank %d, local proc %d, global proc %d" % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        init_processes(0, size, train, args)
