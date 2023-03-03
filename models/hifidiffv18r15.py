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
from einops import rearrange
from misc.snake import Snake
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
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1) # [T,128]
        return table


class SpectrogramUpsampler(nn.Module):
    def __init__(self, n_mels, periodic=True):
        super().__init__()
        self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
        self.conv2 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])

        if periodic:
            self.act1 = Snake(80)
            self.act2 = Snake(80)
        else:
            self.act1 = torch.nn.PReLU()
            self.act2 = torch.nn.PReLU()

    def forward(self, x):
        # x ==> B, 80, H
        x = torch.unsqueeze(x, 1) # B, 1, 80, H
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = torch.squeeze(x, 1) # B, 80, T
        return x


class ResidualBlock(nn.Module):
    def __init__(self, n_mels, residual_channels, dilation, n_cond_global=None):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(512, residual_channels)
        self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)
        if n_cond_global is not None:
            self.conditioner_projection_global = Conv1d(n_cond_global, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step, conditioner_global=None):
        # x ==> (b d), c, t
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)

        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner

        if conditioner_global is not None:
            y = y + self.conditioner_projection_global(conditioner_global)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return x + residual, skip

class LFResidualBlock(nn.Module):
    def __init__(self, n_mels, residual_channels, dilation, n_cond_global=None):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(512, residual_channels)
        self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)
        if n_cond_global is not None:
            self.conditioner_projection_global = Conv1d(n_cond_global, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)
        #self.snake = Snake(residual_channels)

    def forward(self, x, conditioner, diffusion_step, conditioner_global=None):
        # x ==> (b d), c, t
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)

        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner

        if conditioner_global is not None:
            y = y + self.conditioner_projection_global(conditioner_global)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        #y = self.snake(y)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip

class HFResidualBlock(nn.Module):
    def __init__(self, n_mels, residual_channels, dilation, n_cond_global=None):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(512, residual_channels)
        self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)
        if n_cond_global is not None:
            self.conditioner_projection_global = Conv1d(n_cond_global, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step, conditioner_global=None):
        # x ==> (b d), c, t
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)

        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner

        if conditioner_global is not None:
            y = y + self.conditioner_projection_global(conditioner_global)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip

class HifiDiffV18R15(nn.Module):
    def __init__(self, params):
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
        self.spectrogram_upsampler1 = SpectrogramUpsampler(self.n_mels, periodic=True)
        self.spectrogram_upsampler2 = SpectrogramUpsampler(self.n_mels, periodic=False)

        if self.condition_prior_global:
            self.global_condition_upsampler = SpectrogramUpsampler(self.n_cond)
        self.hf_residual_layers = nn.ModuleList([
            HFResidualBlock(self.n_mels, params.residual_channels, 2 ** (i % params.dilation_cycle_length // 2),
                          n_cond_global=self.n_cond)
            for i in range(params.residual_layers)
        ])
        self.lf_residual_layers = nn.ModuleList([
            LFResidualBlock(self.n_mels, params.residual_channels, 2 ** (i % (params.dilation_cycle_length)),
                          n_cond_global=self.n_cond)
            for i in range(params.residual_layers)
        ])

        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
        self.output_projection = Conv1d(params.residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

        print('num param: {}'.format(sum(p.numel() for p in self.parameters() if p.requires_grad)))

    def forward(self, audio, spectrogram, diffusion_step, global_cond=None, **kwargs):
        # audio => (b,t)
        # spectrogram => b, 80, t
        #x = audio.unsqueeze(1) 

        with torch.no_grad():
            audio = audio.unsqueeze(1).repeat(1, 2, 1) # (b, 2, t)

        x = rearrange(audio, "b (d c1) t -> (b d) c1 t", c1=1)
        x = self.input_projection(x) # (b d), c, t
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step) # b, t, 512
        spectrogram1 = self.spectrogram_upsampler1(spectrogram) # b, 80, t periodic
        spectrogram2 = self.spectrogram_upsampler2(spectrogram) # b, 80, t not periodic

        if global_cond is not None:
            global_cond = self.global_condition_upsampler(global_cond)

        x = rearrange(x, "(b d) c t -> b d c t", d=2)
        hf_x, lf_x = torch.chunk(x, 2, dim=1) # hf_x => b 1 c t, lf_x => b 1 c t, 
        hf_x = hf_x.squeeze() # b c t
        lf_x = lf_x.squeeze() # b c t

        hf_skips, lf_skips = [], []
        for lf_res, hf_res in zip(self.hf_residual_layers, self.lf_residual_layers):
            lf_x, lf_skip = lf_res(lf_x, spectrogram1, diffusion_step, global_cond)
            hf_x, hf_skip = hf_res(hf_x, spectrogram2, diffusion_step, global_cond)

            lf_skips.append(lf_skip)
            hf_skips.append(hf_skip)

        hf_x = torch.sum(torch.stack(hf_skips), dim=0) / sqrt(len(self.hf_residual_layers)) #b c t
        lf_x = torch.sum(torch.stack(lf_skips), dim=0) / sqrt(len(self.lf_residual_layers)) #b c t
        hf_x = hf_x.unsqueeze(1) # b 1 c t
        lf_x = lf_x.unsqueeze(1) # b 1 c t
        x = torch.cat([hf_x, lf_x], dim=1) # b 2 c t

        x = rearrange(x, "b d c t -> (b d) c t")
        x = self.skip_projection(x)
        x = F.relu(x)

        x = rearrange(x, "(b d) c t -> b d c t", d=2) 
        hf_x, lf_x = torch.chunk(x, 2, dim=1) # hf_x => b 1 c t, lf_x => b 1 c t
        hf_x = hf_x.squeeze(dim=1) # b c t
        lf_x = lf_x.squeeze(dim=1) # b c t

        x = hf_x + lf_x
        x = self.output_projection(x) # b 1 t

        if self.training:
            return x
        else:
            return x, hf_x, lf_x
