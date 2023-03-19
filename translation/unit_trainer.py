#code reference: Ming-Yu Liu and Christian Clauss and et al, https://github.com/mingyuliutw/UNIT

import torch.nn as nn
import torch
from networks import MsImageDis, VAEGen
from helpers import weights_init, get_scheduler
import os


class UNIT_Trainer(nn.Module):
    """unsupervised image-to-image translation"""
    def __init__(self, hyperparameters):
        super(UNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = VAEGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = VAEGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))

    def forward(self, x_a, x_b):
        self.eval()
        h_a, _ = self.gen_a.encode(x_a)
        h_b, _ = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        self.train()
        return x_ab, x_ba

    def recon_loss(self, input, target):
        """ Reconstruction loss"""
        return torch.mean(torch.abs(input - target))

    def kl_loss(self, mu):
        """ KL convergence loss"""
        return torch.mean(mu**2)

    def gen_update(self, x_a, x_b, hyperparameters):
        self.gen_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # translate: cross domain
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # recon: within domain
        x_a_recon = self.gen_a.decode(h_a + n_a)
        x_b_recon = self.gen_b.decode(h_b + n_b)
        # cycle encode
        h_b2, n_b2 = self.gen_a.encode(x_ba)
        h_a2, n_a2 = self.gen_b.encode(x_ab)
        # cycle translate
        x_aba = self.gen_a.decode(h_a2 + n_a2) 
        x_bab = self.gen_b.decode(h_b2 + n_b2)
        # recon loss (recon + kl converge)
        self.loss_gen_recon_x_a = self.recon_loss(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_loss(x_b_recon, x_b)
        self.loss_gen_recon_kl_a = self.kl_loss(h_a)
        self.loss_gen_recon_kl_b = self.kl_loss(h_b)
        # cycle loss (+ kl converge)
        self.loss_gen_cyc_x_a = self.recon_loss(x_aba, x_a)
        self.loss_gen_cyc_x_b = self.recon_loss(x_bab, x_b)
        self.loss_gen_cyc_kl_a = self.kl_loss(h_a2)
        self.loss_gen_cyc_kl_b = self.kl_loss(h_b2)
        # GAN loss
        # (dis_a.calc_gen_loss(input_fake))
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * (self.loss_gen_adv_a + self.loss_gen_adv_b) + \
                              hyperparameters['recon_x_w'] * (self.loss_gen_recon_x_a + self.loss_gen_recon_x_b) + \
                              hyperparameters['recon_kl_w'] * (self.loss_gen_recon_kl_a + self.loss_gen_recon_kl_b) + \
                              hyperparameters['recon_x_cyc_w'] * (self.loss_gen_cyc_x_a + self.loss_gen_cyc_x_b) + \
                              hyperparameters['recon_kl_cyc_w'] * (self.loss_gen_cyc_kl_a + self.loss_gen_cyc_kl_b)
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # translate: cross domain
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # dis loss
        # dis_a.calc_dis_loss(input_fake, input_real)
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * (self.loss_dis_a + self.loss_dis_b)
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def sample_translate(self, x_a, x_b):
        self.eval()
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
            h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(h_a))
            x_b_recon.append(self.gen_b.decode(h_b))
            x_ba.append(self.gen_a.decode(h_b))
            x_ab.append(self.gen_b.decode(h_a))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba, x_ab = torch.cat(x_ba), torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba


    def update_learning_rate(self):
        self.dis_scheduler.step()
        self.gen_scheduler.step()

    def save(self, dire, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(dire, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(dire, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(dire, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)