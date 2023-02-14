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

class FiLM(nn.Module):
  def __init__(self, input_dim, output_dim):
    super().__init__()
    
    self.input_conv = nn.Conv1d(input_dim, input_dim, 3, padding=1)
    self.output_conv = nn.Conv1d(input_dim, output_dim, 3, padding=1)
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.xavier_uniform_(self.input_conv.weight)
    nn.init.xavier_uniform_(self.output_conv.weight)
    nn.init.zeros_(self.input_conv.bias)
    nn.init.zeros_(self.output_conv.bias)

  def forward(self, spectrogram):
    # spectrogram ==> B, C, L
    # diffusion_step ==> B, C, 1
    #spectrogram = spectrogram + diffusion_step # B, C, L
    spectrogram = self.input_conv(spectrogram) # B, C, L
    spectrogram = F.leaky_relu(spectrogram, 0.2)
    #z = self.encoding(z, noise_scale)
    shift, scale = torch.chunk(self.output_conv(spectrogram), 2, dim=1) # B, 4C, L

    return shift, scale # shift ==> B, 2C, L, scale ==> B, 2C, L

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

class SpectrogramUpsampler(nn.Module):
    def __init__(self, n_mels):
        super().__init__()
        self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
        self.conv2 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        x = torch.squeeze(x, 1)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, n_mels, residual_channels, dilation, n_cond_global=None):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        
        self.diffusion_projection = Linear(512, residual_channels)
        self.conditioner_projection = Conv1d(n_mels, residual_channels, 1)
        self.film = FiLM(residual_channels, 4 * residual_channels)

        if n_cond_global is not None:
            self.conditioner_projection_global = Conv1d(n_cond_global, 2 * residual_channels, 1)

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step, conditioner_global=None):
        # x ==> B, C, L
        # spectrogram ==> B, M (n_mels), L
        # diffusion_step ==> B

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1) # (B, C, 1)
        conditioner = self.conditioner_projection(conditioner) # B, C, L

        y = x + diffusion_step
        y = self.dilated_conv(x) # B, 2C, L
        film_shift, film_scale = self.film(conditioner) 
        # film_shift ==> B, 2C, L, film_scale ==> B, 2C, L

        #print(f"y: {y.shape}, film_shift: {film_shift.shape}, film_scale: {film_scale.shape}")
        y = film_scale * y + film_shift #B, 2C, L
        
        #if conditioner_global is not None:
        #    y = y + self.conditioner_projection_global(conditioner_global)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip

class HifiDiffV7R1(nn.Module):
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

        self.spectrogram_upsampler = SpectrogramUpsampler(self.n_mels)
        if self.condition_prior_global:
            self.global_condition_upsampler = SpectrogramUpsampler(self.n_cond)
        self.residual_layers = nn.ModuleList([
            ResidualBlock(self.n_mels, params.residual_channels, 2 ** (i % params.dilation_cycle_length),
                          n_cond_global=self.n_cond)
            for i in range(params.residual_layers)
        ])
        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
        self.output_projection = Conv1d(params.residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

        print('num param: {}'.format(sum(p.numel() for p in self.parameters() if p.requires_grad)))
        #self.start = torch.cuda.Event(enable_timing=True)
        #self.end = torch.cuda.Event(enable_timing=True)

    def forward(self, audio, spectrogram, diffusion_step, global_cond=None, **kwargs):
        # audio ==> B, L
        # spectrogram ==> B, M (n_mels), F frame(256)
        # diffusion_step ==> B
        x = audio.unsqueeze(1) # B, 1, L
        x = self.input_projection(x) # B, C, L
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step) # B, 512
        spectrogram = self.spectrogram_upsampler(spectrogram) #B, M (n_mels), L

        #if global_cond is not None:
        #    global_cond = self.global_condition_upsampler(global_cond)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, spectrogram, diffusion_step, global_cond)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        
        return x
