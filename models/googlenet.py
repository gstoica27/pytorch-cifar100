"""google net in pytorch



[1] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.

    Going Deeper with Convolutions
    https://arxiv.org/abs/1409.4842v1
"""

import torch
import torch.nn as nn
from positional_encodings import PositionalEncoding2D


class ConvolutionalSelfAttention(nn.Module):
    def __init__(self, spatial_shape, filter_size, approach_args={'name': '1'}):
        super(ConvolutionalSelfAttention, self).__init__()
        self.spatial_H, self.spatial_W, self.spatial_C = spatial_shape
        self.filter_K = filter_size
        self.filter_size = self.filter_K * self.filter_K
        self.approach_name = approach_args['name']
        self.appraoch_args = approach_args
    
        self.setup_approach()    
        self.name2approach = {
            '1': self.forward_on_approach1,
            '2': self.forward_on_approach2,
            '3': self.forward_on_approach3,
            '4': self.forward_on_approach4
        }

        self.input_mask = self.compute_input_mask()
        if torch.cuda.is_available():
            self.input_mask = self.input_mask.cuda()
    
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
            pass
        else:
            raise ValueError('Invalid Approach type')
    
    def maybe_create_positional_encodings(self):
        if self.appraoch_args.get('pos_emb_dim', 0) > 0:
            self.positional_encodings = PositionalEncoding2D(self.appraoch_args['pos_emb_dim'])
        else:
            self.positional_encodings = None

    def cudaify_module(self):
        self.input_mask = self.input_mask.cuda()
    
    def compute_input_mask(self):
        convs_height = self.spatial_H - self.filter_K + 1
        convs_width = self.spatial_W - self.filter_K + 1
        num_convs = convs_height * convs_width
        input_mask = torch.zeros((num_convs, self.spatial_H, self.spatial_W, 1), dtype=torch.float32)
        conv_idx = 0
        for i in range(convs_height):
            for j in range(convs_width):
                input_mask[conv_idx, i:i+self.filter_K, j:j+self.filter_K, :] = 1.
                conv_idx += 1
        self.num_convs = convs_height * convs_width
        self.convs_height = convs_height
        self.convs_width = convs_width
        return input_mask.unsqueeze(1)
    
    def split_input(self, batch):
        batch_r = batch.unsqueeze(0)                                                                # [B,1,H,W,E]
        X_g = batch_r * (1 - self.input_mask)                                                       # [F,B,H,W,E]
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

        X_l = (batch_r * self.input_mask).reshape(                                                  # [F,B,H,W,E]
            self.num_convs, -1, self.spatial_H * self.spatial_W, self.X_encoding_dim                # [F,B,HW,E]
        )
        X_li = torch.argsort(
            (X_l.sum(-1) != 0).type(torch.float32), 
            dim=-1
        )[:, :, -self.filter_size:].sort(dim=-1)[0]                                                 # [F,B,K^2]
        X_li_flat = X_li.reshape(-1, self.filter_size)                                              # [FB,K^2]
        X_l_flat = X_l.reshape(-1, X_l.shape[2], X_l.shape[3])                                      # [FB,HW,E]
        X_l_flat_ = X_l_flat[torch.arange(X_l_flat.shape[0]).unsqueeze(-1), X_li_flat]              # [FB,K^2,E]
        X_l_ = X_l_flat_.reshape(self.num_convs, -1, self.filter_size, self.X_encoding_dim)         # [F,B,K^2,E]
        return X_l_, X_g_
    
    def cosine_similarity(self, x, y):
        # Assume x, y are [F,B,*,E]
        # pdb.set_trace()
        x_normed = torch.nn.functional.normalize(x, dim=-1)  
        y_normed = torch.nn.functional.normalize(y, dim=-1)
        y_permuted = y_normed.transpose(3, 2)
        res = torch.einsum('ijkl,ijla->ijka', x_normed, y_permuted)
        return res

    def forward_on_approach1(self, batch):
        # batch: [B,H,W,C]
        X_l, X_g = self.split_input(batch)
        W_g = self.global_transform(X_g).sum(2)                                                     # [F,B,(HW-K^2),K^2] -> [F,B,K^2]
        s_g = F.softmax(W_g, dim=-1).unsqueeze(-1)                                                  # [F,B,K^2,1]
        
        X_o = X_l * s_g                                                                             # [F,B,K^2,C]
        output = X_o.sum(2).permute(1, 0, 2)                                                        # [B,F,C]
        output_r = output.reshape(-1, self.convs_height, self.convs_width, self.spatial_C)          # [B,F_H,F_W,C]
        return output_r
    
    def maybe_add_positional_encodings(self, batch):
        if self.positional_encodings is not None:
            positional_encodings = self.positional_encodings(
                batch[:, :, :, :self.appraoch_args['pos_emb_dim']]                                   # [B,H,W,P]
            )
            return torch.cat((batch, positional_encodings), dim=-1)                            # [[B,H,W,C];[B,H,W,P]] -> [B,H,W,C+P]
        return batch
    def forward_on_approach2(self, batch):
        batch_pos = self.maybe_add_positional_encodings(batch)
        X_l, X_g = self.split_input(batch_pos)
        
        X_g_scalars = self.global_transform(X_g)                                                    # [F,B,HW-K^2,1]
        # pdb.set_trace()
        raw_compatibilities = self.cosine_similarity(
            X_g, X_l                                                                                # [F,B,HW-K^2,E], [F,B,K^2,E]
        )                                                                                           # [F,B,HW-K^2,K^2]
        compatabilities = F.softmax(self.appraoch_args['softmax_temp'] * raw_compatibilities, dim=2)
        elem_mul = X_g_scalars * compatabilities                                                    # [F,B,HW-K^2,1] x [F,B,HW-K^2,K^2] -> [F,B,HW-K^2,K^2]
        W_g = elem_mul.sum(dim=2).unsqueeze(-1)                                                     # [F,B,HW-K^2,K^2] -> [F,B,K^2] -> [F,B,K^2,1]
        X_l = X_l[:, :, :, :self.spatial_C]                                                         # [F,B,K^2,E] -> [F,B,K^2,C]
        convolved_X = (W_g * X_l).sum(dim=2).permute(1,0,2)                                         # [F,B,K^2,1] x [F,B,K^2,C] -> [F,B,K^2,C] -> [F,B,C] -> [B,F,C]
        output = convolved_X.reshape(
            -1, self.convs_height, self.convs_width, self.spatial_C
        )                                                                                           # [B,F,C] -> [B,F_H,F_W,C]
        return output

    def forward_on_approach3(self, batch):
        pass

    def forward_on_approach4(self, batch):
        pass

    def forward(self, batch):
        return self.name2approach[self.approach_name](batch)

class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()

        #1x1conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 3x3conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 5x5conv branch
        #we use 2 3x3 conv filters stacked instead
        #of 1 5x5 filters to obtain the same receptive
        #field with fewer parameters
        self.b3 = nn.Sequential(
            nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5, n5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )

        #3x3pooling -> 1x1conv
        #same conv
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(input_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class GoogleNet(nn.Module):

    def __init__(self, num_class=100):
        super().__init__()
        self.prelayer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        #although we only use 1 conv layer as prelayer,
        #we still use name a3, b3.......
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        ##"""In general, an Inception network is a network consisting of
        ##modules of the above type stacked upon each other, with occasional
        ##max-pooling layers with stride 2 to halve the resolution of the
        ##grid"""
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        #input feature size: 8*8*1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.4)
        self.linear = nn.Linear(1024, num_class)

    def forward(self, x):
        x = self.prelayer(x)
        x = self.maxpool(x)
        x = self.a3(x)
        x = self.b3(x)

        x = self.maxpool(x)

        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)

        x = self.maxpool(x)

        x = self.a5(x)
        x = self.b5(x)
        import pdb; pdb.set_trace()
        
        #"""It was found that a move from fully connected layers to
        #average pooling improved the top-1 accuracy by about 0.6%,
        #however the use of dropout remained essential even after
        #removing the fully connected layers."""
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        return x

def googlenet():
    return GoogleNet()


