# Copyright 2022 (c) Microsoft Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt

Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)

class TimeAware_LVCBlock(torch.nn.Module):
    ''' time-aware location-variable convolutions
    '''
    def __init__(self,
                 in_channels,
                 cond_channels,
                 #upsample_ratio,
                 conv_layers=5,
                 conv_kernel_size=3,
                 cond_hop_length=256,
                 kpnet_hidden_channels=64,
                 kpnet_conv_size=3,
                 kpnet_dropout=0.0,
                 noise_scale_embed_dim_out=512
                 ):
        super().__init__()

        self.cond_hop_length = cond_hop_length
        self.conv_layers = conv_layers
        self.conv_kernel_size = conv_kernel_size
        self.convs = torch.nn.ModuleList()

        self.output_projection = Conv1d(in_channels, 2 * in_channels, 1)

        #self.upsample = torch.nn.ConvTranspose1d(in_channels, in_channels,
        #                            kernel_size=upsample_ratio*2, stride=upsample_ratio,
        #                            padding=upsample_ratio // 2 + upsample_ratio % 2,
        #                            output_padding=upsample_ratio % 2)


        self.kernel_predictor = KernelPredictor(
            cond_channels=cond_channels,
            conv_in_channels=in_channels,
            conv_out_channels=2 * in_channels,
            conv_layers=conv_layers,
            conv_kernel_size=conv_kernel_size,
            kpnet_hidden_channels=kpnet_hidden_channels,
            kpnet_conv_size=kpnet_conv_size,
            kpnet_dropout=kpnet_dropout
        )

        # the layer-specific fc for noise scale embedding
        self.fc_t = torch.nn.Linear(noise_scale_embed_dim_out, cond_channels)

        for i in range(conv_layers):
            padding = (3 ** i) * int((conv_kernel_size - 1) / 2)
            conv = torch.nn.Conv1d(in_channels, in_channels, kernel_size=conv_kernel_size, padding=padding, dilation=3 ** i)

            self.convs.append(conv)


    def forward(self, x, c, diffusion_step, global_cond):
        ''' forward propagation of the time-aware location-variable convolutions.
        Args:
            x (Tensor): the input sequence (batch, in_channels, in_length)
            c (Tensor): the conditioning sequence (batch, cond_channels, cond_length)

        Returns:
            Tensor: the output sequence (batch, in_channels, in_length)
        '''
        #x, audio_down, c, noise_embedding = data
        #x, c, noise_embedding = data
        batch, in_channels, in_length = x.shape

        noise = (self.fc_t(diffusion_step)).unsqueeze(-1)  # (B, 80)
        condition = c + noise  # (B, 80, T)
        kernels, bias = self.kernel_predictor(condition)
        #x = F.leaky_relu(x, 0.2)
        #x = self.upsample(x)

        for i in range(self.conv_layers):
            dilation = 2**i 
            #x += audio_down
            #y = F.leaky_relu(x, 0.2)
            #y = self.convs[i](y)
            #y = F.leaky_relu(y, 0.2)

            k = kernels[:, i, :, :, :, :]
            b = bias[:, i, :, :]
            y = self.location_variable_convolution(x, k, b, dilation, self.cond_hop_length)

            gate, filter = torch.chunk(y, 2, dim=1)
            #x = x + torch.sigmoid(y[:, :in_channels, :]) * torch.tanh(y[:, in_channels:, :])
            x = x + torch.sigmoid(gate) * torch.tanh(filter)

        #y = self.output_projection(x)
        #residual, skip = torch.chunk(y, 2, dim=1)
        return x / sqrt(2.0)
        #return x

    def location_variable_convolution(self, x, kernel, bias, dilation, hop_size):
        ''' perform location-variable convolution operation on the input sequence (x) using the local convolution kernl.
        Time: 414 μs ± 309 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each), test on NVIDIA V100.
        Args:
            x (Tensor): the input sequence (batch, in_channels, in_length).
            kernel (Tensor): the local convolution kernel (batch, in_channel, out_channels, kernel_size, kernel_length)
            bias (Tensor): the bias for the local convolution (batch, out_channels, kernel_length)
            dilation (int): the dilation of convolution.
            hop_size (int): the hop_size of the conditioning sequence.
        Returns:
            (Tensor): the output sequence after performing local convolution. (batch, out_channels, in_length).
        '''
        batch, in_channels, in_length = x.shape
        batch, in_channels, out_channels, kernel_size, kernel_length = kernel.shape

        assert in_length == (kernel_length * hop_size), "length of (x, kernel) is not matched"

        padding = dilation * int((kernel_size - 1) / 2)
        x = F.pad(x, (padding, padding), 'constant', 0)  # (batch, in_channels, in_length + 2*padding)
        x = x.unfold(2, hop_size + 2 * padding, hop_size)  # (batch, in_channels, kernel_length, hop_size + 2*padding)

        if hop_size < dilation:
            x = F.pad(x, (0, dilation), 'constant', 0)
        x = x.unfold(3, dilation,
                     dilation)  # (batch, in_channels, kernel_length, (hop_size + 2*padding)/dilation, dilation)
        x = x[:, :, :, :, :hop_size]
        x = x.transpose(3, 4)  # (batch, in_channels, kernel_length, dilation, (hop_size + 2*padding)/dilation)
        x = x.unfold(4, kernel_size, 1)  # (batch, in_channels, kernel_length, dilation, _, kernel_size)

        o = torch.einsum('bildsk,biokl->bolsd', x, kernel)
        o = o + bias.unsqueeze(-1).unsqueeze(-1)
        o = o.contiguous().view(batch, out_channels, -1)
        return o

class KernelPredictor(torch.nn.Module):
    ''' Kernel predictor for the time-aware location-variable convolutions
    '''

    def __init__(self,
                 cond_channels,
                 conv_in_channels,
                 conv_out_channels,
                 conv_layers,
                 conv_kernel_size=3,
                 kpnet_hidden_channels=64,
                 kpnet_conv_size=3,
                 kpnet_dropout=0.0,
                 kpnet_nonlinear_activation="LeakyReLU",
                 kpnet_nonlinear_activation_params={"negative_slope": 0.1}
                 ):
        '''
        Args:
            cond_channels (int): number of channel for the conditioning sequence,
            conv_in_channels (int): number of channel for the input sequence,
            conv_out_channels (int): number of channel for the output sequence,
            conv_layers (int):
            kpnet_
        '''
        super().__init__()

        self.conv_in_channels = conv_in_channels
        self.conv_out_channels = conv_out_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_layers = conv_layers

        l_w = conv_in_channels * conv_out_channels * conv_kernel_size * conv_layers
        l_b = conv_out_channels * conv_layers

        padding = (kpnet_conv_size - 1) // 2
        self.input_conv = torch.nn.Sequential(
            torch.nn.Conv1d(cond_channels, kpnet_hidden_channels, 5, padding=(5 - 1) // 2, bias=True),
            getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
        )

        self.residual_conv = torch.nn.Sequential(
            torch.nn.Dropout(kpnet_dropout),
            torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True),
            getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
            torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True),
            getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
            torch.nn.Dropout(kpnet_dropout),
            torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True),
            getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
            torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True),
            getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
            torch.nn.Dropout(kpnet_dropout),
            torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True),
            getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
            torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True),
            getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
        )

        self.kernel_conv = torch.nn.Conv1d(kpnet_hidden_channels, l_w, kpnet_conv_size,
                                           padding=padding, bias=True)
        self.bias_conv = torch.nn.Conv1d(kpnet_hidden_channels, l_b, kpnet_conv_size, padding=padding,
                                         bias=True)

    def forward(self, c):
        '''
        Args:
            c (Tensor): the conditioning sequence (batch, cond_channels, cond_length)
        Returns:
        '''
        batch, cond_channels, cond_length = c.shape

        c = self.input_conv(c)
        c = c + self.residual_conv(c)
        k = self.kernel_conv(c)
        b = self.bias_conv(c)

        kernels = k.contiguous().view(batch,
                                      self.conv_layers,
                                      self.conv_in_channels,
                                      self.conv_out_channels,
                                      self.conv_kernel_size,
                                      cond_length)
        bias = b.contiguous().view(batch,
                                   self.conv_layers,
                                   self.conv_out_channels,
                                   cond_length)
        return kernels, bias

class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        self.projection1 = Linear(128, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)  # [1,64]
        table = steps * 10.0 ** (dims * 4.0 / 63.0)  # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class HifiDiffV3(nn.Module):
    def __init__(self, params=None):
        super().__init__()
        self.params = params
        self.use_prior = params.use_prior
        self.condition_prior = params.condition_prior
        self.condition_prior_global = params.condition_prior_global

        assert not (self.condition_prior and self.condition_prior_global),\
          "use only one option for conditioning on the prior"
        
        print("use_prior: {}".format(self.use_prior))
        self.n_mels = params.n_mels
        self.n_cond = None

        print("condition_prior: {}".format(self.condition_prior))
        if self.condition_prior:
            self.n_mels = self.n_mels + 1
            print("self.n_mels increased to {}".format(self.n_mels))
        
        print("condition_prior_global: {}".format(self.condition_prior_global))
        if self.condition_prior_global:
            self.n_cond = 1

        self.input_projection = Conv1d(1, params.residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))
        #self.spectrogram_upsampler = SpectrogramUpsampler(self.n_mels)
        #if self.condition_prior_global:
        #    self.global_condition_upsampler = SpectrogramUpsampler(self.n_cond)
        
        #self.residual_layers = nn.ModuleList([
        #    ResidualBlock(self.n_mels, params.residual_channels, 2 ** (i % params.dilation_cycle_length),
        #                  n_cond_global=self.n_cond)
        #    for i in range(params.residual_layers)
        #])

        self.residual_layers = nn.ModuleList()

        inner_channels = params.residual_channels
        cond_hop_length = params.hop_samples
        cond_channels = params.n_mels

        for _ in range(params.residual_layers):
            #cond_hop_length = cond_hop_length * upsample_ratios[i]
            lvcb = TimeAware_LVCBlock(
                in_channels=inner_channels,
                cond_channels=cond_channels,
                #upsample_ratio=upsample_ratios[i],
                conv_layers=params.lvc_layers_each_block,
                conv_kernel_size=params.lvc_kernel_size,
                cond_hop_length=cond_hop_length,
                kpnet_hidden_channels=params.kpnet_hidden_channels,
                kpnet_conv_size=params.kpnet_conv_size,
                kpnet_dropout=params.kpnet_dropout,
                #noise_scale_embed_dim_out=diffusion_step_embed_dim_out
            )

            self.residual_layers += [lvcb]
            #self.downsample.append(DiffusionDBlock(inner_channels, inner_channels, upsample_ratios[self.lvc_block_nums-n-1]))

        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
        self.output_projection = Conv1d(params.residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

        print('num param: {}'.format(sum(p.numel() for p in self.parameters() if p.requires_grad)))

    def forward(self, audio, spectrogram, diffusion_step, global_cond=None):
        x = audio.unsqueeze(1)
        x = self.input_projection(x)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        #spectrogram = self.spectrogram_upsampler(spectrogram)
        #if global_cond is not None:
        #    global_cond = self.global_condition_upsampler(global_cond)

        skip = []
        for layer in self.residual_layers:
            x = layer(x, spectrogram, diffusion_step, global_cond)
            #skip.append(skip_connection)

        #x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        #x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x
