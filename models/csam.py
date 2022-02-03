from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import pdb
from tqdm import tqdm
from positional_encodings import PositionalEncoding2D
"""
Modified from: https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

class ConvolutionalSelfAttention(nn.Module):
    def __init__(self,
        spatial_shape,
        filter_size,
        approach_args={'name': '4', 'padding': 'valid', 'stride': 1}
    ):
        super(ConvolutionalSelfAttention, self).__init__()
        self.spatial_H, self.spatial_W, self.spatial_C = spatial_shape
        self.apply_stochastic_stride = approach_args['apply_stochastic_stride']
        self.stride = approach_args.get('stride', 1)
        self.filter_K = filter_size
        self.filter_size = self.filter_K * self.filter_K
        self.approach_name = approach_args['name']
        self.appraoch_args = approach_args
        self.padding_type = approach_args.get('padding', 'valid')
        self.input_padder = self.compute_padding(self.padding_type)

        self.setup_approach()
        self.name2approach = {
            '1': self.efficient_approach1,
            '2': self.efficient_approach2,
            '3': self.forward_on_approach3,
            '4': self.forward_on_approach4
        }

        self.local_mask = self.compute_input_mask()
        if self.approach_name in {'3', '4'}:
            self.local_mask = self.local_mask.reshape(self.num_convs, 1, self.spatial_H, self.spatial_W, 1)
        if torch.cuda.is_available():
            self.local_mask = self.local_mask.cuda()
            self.local_indices = self.local_indices.cuda()

    def compute_padding(self, padding_type):
        if padding_type.lower() == 'valid':
            padding_tuple = (0, 0, 0, 0)
        elif padding_type.lower() == 'same':
            padding_x = int((self.stride * (self.spatial_W - 1) - self.spatial_W + self.filter_K) / 2)
            padding_y = int((self.stride * (self.spatial_H - 1) - self.spatial_H + self.filter_K) / 2)
            padding_tuple = (padding_x, padding_x, padding_y, padding_y)
        else:
            raise ValueError('Unknown padding type requested')
        self.padding_tuple = padding_tuple
        input_padder = nn.ConstantPad2d(padding_tuple, 0.)
        return input_padder

    def get_output_shape(self):
        h_dim = (self.spatial_H - self.filter_K + 2 * self.padding_tuple[2]) / self.stride
        w_dim = (self.spatial_W - self.filter_K + 2 * self.padding_tuple[0]) / self.stride
        return [h_dim, w_dim, self.spatial_C]

    def setup_approach(self):
        self.X_encoding_dim = self.spatial_C                                                        # Call this E
        # Account for positional encodings
        self.maybe_create_positional_encodings()
        if self.appraoch_args.get('pos_emb_dim', 0) > 0:
            self.X_encoding_dim += self.appraoch_args['pos_emb_dim']
        if self.approach_name == '1':
            self.global_transform = nn.Linear(self.X_encoding_dim, self.filter_K * self.filter_K)
        elif self.approach_name == '2':
            self.global_transform = nn.Linear(self.X_encoding_dim, 1)
        elif self.approach_name == '3':
            self.global_transform = nn.Linear(self.X_encoding_dim, self.spatial_C)
        elif self.approach_name == '4':
            self.key_transform = nn.Linear(self.X_encoding_dim, self.X_encoding_dim)
            self.query_transform = nn.Linear(self.X_encoding_dim, self.X_encoding_dim)
            self.value_transform = nn.Linear(self.X_encoding_dim, 1)
        else:
            raise ValueError('Invalid Approach type')

    def maybe_create_positional_encodings(self):
        if self.appraoch_args.get('pos_emb_dim', 0) > 0:
            self.positional_encodings = PositionalEncoding2D(self.appraoch_args['pos_emb_dim'])
        else:
            self.positional_encodings = None

    def cudaify_module(self):
        self.local_mask = self.local_mask.cuda()
        self.local_indices = self.local_indices()

    def compute_cell_lengths(self, num_cells, total_length):
        cell_lengths = self.stride * torch.ones(num_cells, dtype=torch.int)

        # convs_height and convs_width are not necessarily divisible by stride
        if self.apply_stochastic_stride:
            leftover_length = total_length - self.stride * num_cells
            for _ in range(leftover_length):
                i = random.randrange(len(cell_lengths))
                cell_lengths[i] += 1

        return cell_lengths

    def compute_input_mask(self):
        convs_height = self.spatial_H - self.filter_K + 1 + self.padding_tuple[2]
        convs_width = self.spatial_W - self.filter_K + 1 + self.padding_tuple[0]
        num_convs = convs_height * convs_width

        strided_convs_height = convs_height // self.stride
        strided_convs_width = convs_width // self.stride
        num_strided_convs = strided_convs_height * strided_convs_width

        cell_heights = self.compute_cell_lengths(strided_convs_height, convs_height)
        cell_widths = self.compute_cell_lengths(strided_convs_width, convs_width)

        cell_row_starts = torch.cumsum(torch.cat((torch.zeros(1, dtype=torch.int), cell_heights[:-1])), dim=0)
        cell_col_starts = torch.cumsum(torch.cat((torch.zeros(1, dtype=torch.int), cell_widths[:-1])), dim=0)

        input_mask = torch.zeros(
            (num_strided_convs, self.spatial_H + 2 * self.padding_tuple[2], self.spatial_W + 2*self.padding_tuple[0]),
            dtype=torch.float32)
        self.local_indices = torch.zeros((num_strided_convs, self.filter_K * self.filter_K))
        conv_idx = 0
        for i_idx, i in enumerate(cell_row_starts):
            for j_idx, j in enumerate(cell_col_starts):
                offset_i = i + (random.randrange(cell_heights[i_idx]) if self.apply_stochastic_stride else 0)
                offset_j = j + (random.randrange(cell_widths[j_idx]) if self.apply_stochastic_stride else 0)
                input_mask[conv_idx, offset_i:offset_i+self.filter_K, offset_j:offset_j+self.filter_K] = 1.
                self.local_indices[conv_idx] = input_mask[conv_idx].reshape(-1).nonzero().sort()[0].reshape(-1)
                conv_idx += 1
        self.num_convs = strided_convs_height * strided_convs_width
        self.convs_height = strided_convs_height
        self.convs_width = strided_convs_width
        return input_mask.flatten(1) # [Nc, HW]

    def split_input(self, batch, mask_X_g=True):
        batch_r = batch.unsqueeze(0)                                                                    # [1,B,H,W,E]
        if mask_X_g:
            X_g = batch_r * (1 - self.local_mask)                                                       # [F,B,H,W,E]

            X_g = X_g.reshape(self.num_convs, -1, self.spatial_H * self.spatial_W, self.X_encoding_dim) # [F,B,HW,E]
            X_gi = torch.argsort(
                (X_g.sum(-1) != 0).type(torch.float32),
                dim=-1
            )[:, :, self.filter_size:].sort(dim=-1)[0]                                                  # [F,B,(HW-K^2)]
            X_g_flat = X_g.reshape(-1, X_g.shape[2], X_g.shape[3])                                      # [FB,HW,E]
            X_gi_flat = X_gi.reshape(-1, X_gi.shape[2])                                                 # [FB,(HW-K^2)]
            X_g_flat_ = X_g_flat[torch.arange(X_g_flat.shape[0]).unsqueeze(-1), X_gi_flat]              # [FB,(HW-K^2),E]
            X_g_ = X_g_flat_.reshape(
                self.num_convs, -1, X_g_flat_.shape[1], X_g_flat_.shape[2]
            )
        else:
            X_g_ = batch_r.reshape(1, -1, self.spatial_H * self.spatial_W, self.X_encoding_dim)         # [1,B,HW,E]
        X_l = (batch_r * self.local_mask).reshape(                                                      # [F,B,H,W,E]
            self.num_convs, -1, self.spatial_H * self.spatial_W, self.X_encoding_dim                    # [F,B,HW,E]
        )
        X_li = torch.argsort(
            (X_l.sum(-1) != 0).type(torch.float32),
            dim=-1
        )[:, :, -self.filter_size:].sort(dim=-1)[0]                                                     # [F,B,K^2]
        X_li_flat = X_li.reshape(-1, self.filter_size)                                                  # [FB,K^2]
        X_l_flat = X_l.reshape(-1, X_l.shape[2], X_l.shape[3])                                          # [FB,HW,E]
        X_l_flat_ = X_l_flat[torch.arange(X_l_flat.shape[0]).unsqueeze(-1), X_li_flat]                  # [FB,K^2,E]
        X_l_ = X_l_flat_.reshape(self.num_convs, -1, self.filter_size, self.X_encoding_dim)             # [F,B,K^2,E]
        return X_l_, X_g_

    def cosine_similarity(self, x, y):
        # Assume x, y are [F,B,*,E]
        x_normed = torch.nn.functional.normalize(x, dim=-1)
        y_normed = torch.nn.functional.normalize(y, dim=-1)
        if len(x.shape) == 3:
            y_permuted = y_normed.transpose(2, 1)
            res = torch.bmm(x_normed, y_permuted)
        else:
            y_permuted = y_normed.transpose(3, 2)
            res = torch.einsum('ijkl,ijla->ijka', x_normed, y_permuted)
        return res

    def efficient_approach1(self, batch):
        batch = self.maybe_add_positional_encodings(batch)
        batch_flat = batch.flatten(1,2)                                                             # [B,H,W,C] -> [B,HW,C]
        gX = self.global_transform(batch_flat)                                                      # [B,HW,C] -> [B,HW,K^2]
        global_mask = 1 - self.local_mask                                                           # [Nc,HW]
        convolutions = torch.einsum(
            'ijk,kl->ijl',
            gX.transpose(2, 1),                                                                     # [B,HW,K^2] -> [B,K^2,HW]
            global_mask.transpose(1,0)                                                              # [Nc,HW] -> [HW,Nc]
            ).transpose(2,1)                                                                        # [B,K^2,HW]x[HW,Nc] -> [B,K^2,Nc] -> [B,Nc,K^2]
        local_idxs_onehot = F.one_hot(self.local_indices.type(torch.int64))                         # OneHot([Nc,HW]) -> [Nc,K^2,HW]
        Xl = torch.einsum('ijk,lmj->ilmk', batch_flat, local_idxs_onehot.type(torch.float))         # [B,HW,C],[Nc,K^2,HW] -> [B,Nc,K^2,C]
        result = torch.einsum('abcd,abc->abd', Xl, convolutions)                                    # [B,Nc,K^2,C]x[B,Nc,K^2] -> [B,Nc,C]
        return result[:,:,:self.spatial_C].reshape(
            -1, self.convs_height, self.convs_width, self.spatial_C
            )                                                                                       # [B,F,F,C]

    def maybe_add_positional_encodings(self, batch):
        if self.positional_encodings is not None:
            positional_encodings = self.positional_encodings(
                batch[:, :, :, :self.appraoch_args['pos_emb_dim']]                                   # [B,H,W,P]
            )
            return torch.cat((batch, positional_encodings), dim=-1)                            # [[B,H,W,C];[B,H,W,P]] -> [B,H,W,C+P]
        return batch

    def efficient_approach2(self, batch):
        batch_pos = self.maybe_add_positional_encodings(batch)
        global_mask = 1 - self.local_mask                                                           # [Nc, HW]
        batch_flat = batch_pos.flatten(1, 2)                                                        # [B,H,W,C] -> [B,HW,C]
        gX = self.global_transform(batch_flat)                                                      # [B,HW,C] -> [B,HW,1]
        cos_sim = self.cosine_similarity(batch_flat, batch_flat)                                    # [B,HW,C]x[B,HW,C] -> [B,HW,HW]
        exp_sim = torch.exp(cos_sim - cos_sim.mean(dim=-1, keepdims=True)[0])                       # [B,HW,HW]
        exp_sum = torch.einsum('ijk,kl->ijl', exp_sim, global_mask.transpose(1, 0))                 # [B,HW,HW]x[HW,Nc] -> [B,HW,Nc]
        inverted_sum = 1. / exp_sum                                                                 # [B,HW,Nc]
        masked_denominator = inverted_sum * self.local_mask.transpose(1,0).unsqueeze(0)             # [B,HW,Nc]x([Nc,Hw] -> [HW,Nc] -> [1,HW,Nc]) -> [B,HW,Nc]
        g_sim = gX.transpose(2, 1) * exp_sim                                                        # ([B,HW,1] -> [B,1,HW])x[B,HW,HW] -> [B,HW,HW]
        g_sum = torch.einsum('ijk,kl->ijl', g_sim, global_mask.transpose(1, 0))                     # [B,HW,HW]x([B,Nc,HW] -> [B,HW,Nc]) -> [B,HW,Nc]
        extended_filter = g_sum * masked_denominator                                                # [B,HW,Nc]
        result = torch.bmm(batch_flat.transpose(2, 1), extended_filter).transpose(2, 1)             # ([B,HW,C] -> [B,C,HW])x[B,HW,Nc] -> [B,C,Nc] -> [B,Nc,C]

        return result[:,:,:self.spatial_C].reshape(
            -1, self.convs_height, self.convs_width, self.spatial_C
        )                                                                                           # [B,Nc,C] -> [B,F,F,C]

    def forward_on_approach3(self, batch):
        batch_pos = self.maybe_add_positional_encodings(batch)
        X_l, X_g = self.split_input(batch_pos)

        X_g_vectors = self.global_transform(X_g)                                                    # [F,B,HW-K^2,C]

        raw_compatibilities = self.cosine_similarity(
            X_g, X_l                                                                                # [F,B,HW-K^2,C], [F,B,K^2,C]
        )                                                                                           # [F,B,HW-K^2,K^2]
        compatabilities = F.softmax(self.appraoch_args['softmax_temp'] * raw_compatibilities, dim=2)

        W_g = torch.einsum('FBGE,FBGL->FBLE', X_g_vectors, compatabilities)                         # [F,B,K^2,C]
        X_l = X_l[:, :, :, :self.spatial_C]                                                         # [F,B,K^2,E] -> [F,B,K^2,C]
        forget_gate = torch.sigmoid(torch.sum(W_g * X_l, dim=-1)).unsqueeze(-1)                     # [F,B,K^2,1]

        convolved_X = (forget_gate * X_l).sum(dim=2).permute(1,0,2)                                 # [F,B,K^2,1] x [F,B,K^2,C] -> [F,B,K^2,C] -> [F,B,C] -> [B,F,C]
        output = convolved_X.reshape(
            -1, self.convs_height, self.convs_width, self.spatial_C
        )                                                                                           # [B,F,C] -> [B,F_H,F_W,C]
        return output

    def forward_on_approach4(self, batch):
        batch_pos = self.maybe_add_positional_encodings(batch)
        X_l, X_g = self.split_input(batch_pos, mask_X_g=False)

        keys = self.key_transform(X_g)                                                              # [1,B,HW,E]
        queries = self.query_transform(X_l)                                                         # [F,B,K^2,E]
        values = self.value_transform(X_g)                                                          # [1,B,HW,1]

        raw_compatibilities = self.cosine_similarity(
            keys, queries                                                                           # [1,B,HW,E], [F,B,K^2,E]
        )                                                                                           # [F,B,HW,K^2]
        compatabilities = F.softmax(
            self.appraoch_args['softmax_temp'] * raw_compatibilities,
            dim=2
        )
        elem_mul = values * compatabilities                                                         # [1,B,HW,1] x [F,B,HW,K^2] -> [F,B,HW,K^2]
        W_g = elem_mul.sum(dim=2).unsqueeze(-1)                                                     # [F,B,HW,K^2] -> [F,B,K^2] -> [F,B,K^2,1]
        X_l = X_l[:, :, :, :self.spatial_C]                                                         # [F,B,K^2,E] -> [F,B,K^2,C]
        convolved_X = (W_g * X_l).sum(dim=2).permute(1,0,2)                                         # [F,B,K^2,1] x [F,B,K^2,C] -> [F,B,K^2,C] -> [F,B,C] -> [B,F,C]
        output = convolved_X.reshape(
            -1, self.convs_height, self.convs_width, self.spatial_C
        )                                                                                           # [B,F,C] -> [B,F_H,F_W,C]
        return output

    def forward(self, batch):
        """
        Input shape expected to be [B,C,H,W]
        """
        batch = self.input_padder(batch)                                                            # Pad batch for resolution reduction/preservation
        batch = batch.permute(0, 2, 3, 1)                                                           # [B,C,H,W] -> [B,H,W,C]
        output = self.name2approach[self.approach_name](batch)                                      # [B,C,F,F] -> [B,F,F,C]
        output = output.permute(0, 3, 1, 2)                                                         # [B,F,F,C] -> [B,C,F,F]
        return output

class ConvAttnWrapper(nn.Module):
    def __init__(self, backbone, variant_kwargs):
        super().__init__()
        self.variant_kwargs = variant_kwargs
        # instantiate the backbone
        self.backbone = backbone
        # Obtain ordered list of backbone layers | Layer spatial information
        self.backbone_layers, self.backbone_spatial_shapes = self.backbone.get_network_structure()

        self.network_structure = self.inject_variant()

    def inject_variant(self):
        injection_instructions = self.variant_kwargs['injection_info']
        network_structure = []
        start_point = 0
        for injection_instruction in injection_instructions:
            inject_layer, inject_number, filter_size = injection_instruction
            # Add backbone layers up to injection point
            network_structure += self.backbone_layers[start_point: inject_layer]
            spatial_shape = self.backbone_spatial_shapes[inject_layer-1]
            # Stack network modules
            for i in range(inject_number):
                variant_module = ConvolutionalSelfAttention(
                    spatial_shape=spatial_shape,
                    filter_size=filter_size,
                    approach_args=self.variant_kwargs
                )
                variant_bn = nn.BatchNorm2d(spatial_shape[-1])
                network_structure += [variant_module, variant_bn]
                spatial_shape = variant_module.get_output_shape()

            start_point = inject_layer
        network_structure += self.backbone_layers[start_point:]
        network_structure = nn.Sequential(*network_structure)
        return network_structure

    def forward(self, batch):
        return self.network_structure(batch)
