import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings import PositionalEncodingPermute2D


def is_nan(x, enforce=False):
    if not enforce: return
    nans = x.isnan().sum()
    infs = x.isinf().sum()
    flag = nans + infs
    if flag:
        print(nans, infs)
        pdb.set_trace()

class LocalWindowSelfAttention(nn.Module):
    def __init__(
        self, spatial_shape, window_size, num_heads, qkv_bias=True, 
        qk_scale=None, attn_drop=0., proj_drop=0.,  padding='same', 
        stride=1
    ):
        super().__init__()

        self.input_H, self.input_W, self.input_C = spatial_shape
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv_bias =qkv_bias
        self.qk_scale = qk_scale

        self.padding = padding
        self.stride = stride

        self.position_bias = PositionalEncodingPermute2D(self.input_C)
        self.add_padding, self.padding_mask, self.padding_amounts = self.compute_padding(self.padding)
        self.input_H += self.padding_amounts[2] + self.padding_amounts[3]
        self.input_W += self.padding_amounts[0] + self.padding_amounts[1]
        self.window_mask = self.compute_window_mask()
        if torch.cuda.is_available:
            self.window_mask = self.window_mask.cuda()

        self.kv_proj = nn.Linear(self.input_C, 2 * self.input_C, bias=qkv_bias)
        self.q_proj = nn.Linear(self.input_C, self.input_C)
        self.attn_drop = nn.Dropout(attn_drop)
        self.output_proj = nn.Linear(self.input_C, self.input_C)
        self.output_drop = nn.Dropout(proj_drop)
        

    def compute_padding(self, padding_type):
        if padding_type.lower() == 'valid':
            padding_amounts = (0, 0, 0, 0)
        elif padding_type.lower() == 'same':
            padding_x = int((self.stride * (self.input_W - 1) - self.input_W + self.window_size) / 2)
            padding_y = int((self.stride * (self.input_H - 1) - self.input_H + self.window_size) / 2)
            padding_amounts = (padding_x, padding_x, padding_y, padding_y)
        else:
            raise ValueError('Unknown padding type requested')
        padding_amounts = padding_amounts
        add_padding = nn.ConstantPad2d(padding_amounts, 0.)
        # Compute padding indicator mask
        padding_mask = torch.ones((self.input_H, self.input_W))
        padding_mask = 1 - add_padding(padding_mask.unsqueeze(0)).squeeze(0)
        return add_padding, padding_mask, padding_amounts
    
    def compute_cell_lengths(self, num_cells):
        cell_lengths = self.stride * torch.ones(num_cells, dtype=torch.int)

        return cell_lengths

    def compute_window_mask(self):
        unit_stride_convs_height = self.input_H - self.window_size + 1
        unit_stride_convs_width = self.input_W - self.window_size + 1

        self.convs_height = int(unit_stride_convs_height // self.stride)
        self.convs_width = int(unit_stride_convs_width // self.stride)
        self.num_convs = self.convs_height * self.convs_width

        cell_heights = self.compute_cell_lengths(self.convs_height)
        cell_widths = self.compute_cell_lengths(self.convs_width)

        cell_row_starts = torch.cumsum(torch.cat((torch.zeros(1, dtype=torch.int), cell_heights[:-1])), dim=0)
        cell_col_starts = torch.cumsum(torch.cat((torch.zeros(1, dtype=torch.int), cell_widths[:-1])), dim=0)

        input_mask = torch.zeros(
            (
                self.num_convs,
                self.input_H,
                self.input_W,
            ),
            dtype=torch.float32)

        conv_idx = 0
        for i in cell_row_starts:
            for j in cell_col_starts:
                offset_i = i
                offset_j = j
                try:
                    input_mask[conv_idx, offset_i:offset_i+self.window_size, offset_j:offset_j+self.window_size] = 1.
                    input_mask[conv_idx] = F.relu(input_mask[conv_idx] - self.padding_mask)
                except:
                    pdb.set_trace()
                conv_idx += 1
        return input_mask.flatten(1)                                                                                    # [Nc,HW]

    def add_position_encodings(self, batch):
        position_encodings = self.position_bias(batch)
        return position_encodings * math.sqrt(self.input_C) + batch

    def forward(self, batch, query, apply_window_mask=False, add_pos_enc=False, eps=1e-7):
        B, C, _, _= batch.shape
        if add_pos_enc:
            batch = self.add_position_encodings(batch)
        padded_batch = self.add_padding(batch)
        flat_batch = padded_batch.flatten(2)
        _, _, pHpW = flat_batch.shape
        
        kv = self.kv_proj(
            flat_batch.transpose(2, 1)                                                                                  # [B,pHpW,C]
        ).\
            reshape(B, pHpW, 2, self.num_heads, int(C / self.num_heads)).\
                permute(2, 0, 3, 1, 4)                                                                                  # [B,pHpW,2C] -> 
                                                                                                                        #   [B,pHpW,2,nh,C/nh] -> 
                                                                                                                        #     [2,B,nh,pHpW,C/nh]
        k, v = kv[0], kv[1]                                                                                             # [B,nh,pHpW,C/nh] x2
        q = self.q_proj(query).reshape(B, query.shape[1], self.num_heads, int(C /self.num_heads)).transpose(2, 1)            # [B,Q,C] ->
                                                                                                                        #   [B,Q,nh,C/nh] -> 
                                                                                                                        #     [B,nh,Q,C/nh]
        q = q * self.qk_scale
        scores = torch.matmul(q, k.transpose(-2, -1))                                                                   # [B,nh,Q,C/nh] x [B,nh,C/nh,pHpW] -> [B,nh,Q,pHpW]
        is_nan(scores, batch.shape[0] > 1)
        if apply_window_mask:
            # if batch.shape[0] > 1 and q.shape[2] > 1: pdb.set_trace()
            valid_scores = scores - (1 - self.window_mask.unsqueeze(0).unsqueeze(0)) * 1e27                            # [B,nh,Q,pHpW] x [1,1,Nc,pHpW] -> [B,nh,Q\Nc,pHpW]
            is_nan(valid_scores, batch.shape[0] > 1)
            stable_exp = torch.exp(valid_scores - valid_scores.max(-1, keepdim=True)[0])
            is_nan(stable_exp, batch.shape[0] > 1)
            stable_sum = (stable_exp * self.window_mask.unsqueeze(0).unsqueeze(0)).sum(-1)                        # [B,nh,Q,pHpW] x [1,1,Nc,pHpW] -> [B,nh,Nc,pHpW] -> [B,nh,Nc]
            is_nan(stable_sum, batch.shape[0] > 1)
            partial_softmax = stable_exp / stable_sum.unsqueeze(-1)                                                     # [B,nh,Q,pHpW] x [B,nh,Nc,1] -> [B,nh,Nc,pHpW]
            is_nan(partial_softmax, batch.shape[0] > 1)
            masked_softmax = partial_softmax * self.window_mask.unsqueeze(0).unsqueeze(0)                               # [B,nh,Nc,pHpW] x [1,1,Nc,pHpW] -> [B,nh,Nc,pHpW]
            is_nan(masked_softmax, batch.shape[0] > 1)
            masked_softmax = self.attn_drop(masked_softmax)
            is_nan(masked_softmax, batch.shape[0] > 1)
            permuted_softmax = masked_softmax                                                                           # [B,nh,Nc,pHpW] -> [Nc,B,nh,pHpW]
        else:
            # if batch.shape[0] > 1: pdb.set_trace()
            permuted_softmax = F.softmax(scores, dim=-1)                                                                # [B,nh,Q,pHpW]
            is_nan(permuted_softmax, batch.shape[0] > 1)
        # pdb.set_trace()
        # attn_msa_output = torch.einsum('abcd,bcde->abce', permuted_softmax, v)                                          # [B,nh,Q\Nc,pHpW] x [B,nh,pHpW,C/nh] -> [B,nh,Q\Nc,C/nh]
        attn_msa_output = torch.matmul(permuted_softmax, v)                                                             # [B,nh,Q\Nc,pHpW] x [B,nh,pHpW,C/nh] -> [B,nh,Q\Nc,C/nh]
        is_nan(attn_msa_output, batch.shape[0] > 1)
        attn_output = attn_msa_output.transpose(2,1).flatten(2)                                                         # [B,nh,Q\Nc,C/nh] -> [B,Q\Nc,nh,C/nh] -> [B,Q\Nc,C] -> [B,Q\Nc,C]
        is_nan(attn_output, batch.shape[0] > 1)
        
        output = self.output_proj(attn_output)
        is_nan(output, batch.shape[0] > 1)
        output_dropped = self.output_drop(output)
        output_reshaped = output_dropped.transpose(2,1)
        
        if apply_window_mask:
            H, W = self.convs_height, self.convs_width
        else:
            H = W = 1
        
        return output_reshaped.reshape(B, C, H, W)                                                                      # [B,C,H\1,W\1]
    
    def get_output_shape(self):
        return [self.convs_height, self.convs_width, self.input_C]

class BackgroundConditionedMSA(nn.Module):
    def __init__(
        self, spatial_shape, window_size, num_heads, qkv_bias=True, 
        qk_scale=None, attn_drop=0., proj_drop=0., padding='same', 
        stride=1, add_position_encoding=False
    ):
        super().__init__()

        self.spatial_shape = spatial_shape
        self.input_H, self.input_W, self.input_C = spatial_shape
        self.window_size = window_size
        self.add_position_encoding = add_position_encoding

        self.summary_vector = nn.Parameter(torch.rand((1, self.input_C), requires_grad=True))
        self.summary_msa = LocalWindowSelfAttention(
            spatial_shape=spatial_shape,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            padding=padding,
            stride=stride
        )
        self.window_msa = LocalWindowSelfAttention(
            spatial_shape=spatial_shape,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            padding=padding,
            stride=stride
        )

    def forward(self, batch):
        # if batch.shape[0] > 1: pdb.set_trace()
        faux_batched_query = self.summary_vector.unsqueeze(0).repeat(batch.shape[0], 1, 1)                              # [B,1,C]
        global_summary = self.summary_msa(
            batch, query=faux_batched_query, add_pos_enc=self.add_position_encoding
        )                                                                                                               # [B,C,1,1]
        local_summary = self.summary_msa(
            batch, query=faux_batched_query, add_pos_enc=self.add_position_encoding, apply_window_mask=True
        )                                                                                                               # [B,C,H,W]
        background_summary = global_summary - local_summary                                                             # [B,C,H,W] - [B,C,1,1] -> [B,C,H,W]
        background_summary = background_summary.permute(0, 2, 3, 1).flatten(1,2)                                        # [B,H,W,C] -> [B,HW,C]
        window_summary = self.window_msa(
            batch, query=background_summary, add_pos_enc=self.add_position_encoding, apply_window_mask=True
        )                                                                                                               # [B,C,H,W]
        # if window_summary.shape[0] > 1: pdb.set_trace()
        return window_summary                                                                                           # [B,C,H,W]

    def get_output_shape(self):
        return self.window_msa.get_output_shape()

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
            inject_layer, inject_number, window_size = injection_instruction
            # Add backbone layers up to injection point
            network_structure += self.backbone_layers[start_point: inject_layer]
            spatial_shape = self.backbone_spatial_shapes[inject_layer-1]
            # Stack network modules
            for i in range(inject_number):

                variant_module = BackgroundConditionedMSA(
                    spatial_shape=spatial_shape,
                    window_size=window_size,
                    num_heads=self.variant_kwargs['num_heads'],
                    qkv_bias=self.variant_kwargs['qkv_bias'],
                    qk_scale=self.variant_kwargs['qk_scale'],
                    attn_drop=self.variant_kwargs['attn_drop'],
                    proj_drop=self.variant_kwargs['proj_drop'],
                    padding=self.variant_kwargs['padding'],
                    stride=self.variant_kwargs['stride'],
                    add_position_encoding=self.variant_kwargs['add_position_encoding']
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