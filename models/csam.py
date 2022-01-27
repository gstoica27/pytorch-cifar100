from __future__ import print_function
import argparse
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
    def __init__(self, spatial_shape, filter_size, approach_args={'name': '3'}):
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
        batch_pos = self.maybe_add_positional_encodings(batch)
        X_l, X_g = self.split_input(batch_pos)

        X_g_vectors = self.global_transform(X_g)                                                    # [F,B,HW-K^2,C]

        raw_compatibilities = self.cosine_similarity(
            X_g, X_l                                                                                # [F,B,HW-K^2,C], [F,B,K^2,C]
        )                                                                                           # [F,B,HW-K^2,K^2]
        compatabilities = F.softmax(self.appraoch_args['softmax_temp'] * raw_compatibilities, dim=2)

        W_g = torch.einsum('FBGE,FBGL->FBLE', X_g_vectors, compatabilities)                         # [F,B,K^2,C]
        X_l = X_l[:, :, :, :self.spatial_C]                                                         # [F,B,K^2,E] -> [F,B,K^2,C]
        forget_gate = torch.sigmoid(torch.sum(W_g * X_l), dim=-1).unsqueeze(-1)                     # [F,B,K^2,1]

        convolved_X = (forget_gate * X_l).sum(dim=2).permute(1,0,2)                                 # [F,B,K^2,1] x [F,B,K^2,C] -> [F,B,K^2,C] -> [F,B,C] -> [B,F,C]
        output = convolved_X.reshape(
            -1, self.convs_height, self.convs_width, self.spatial_C
        )                                                                                           # [B,F,C] -> [B,F_H,F_W,C]
        return output

    def forward_on_approach4(self, batch):
        pass

    def forward(self, batch):
        return self.name2approach[self.approach_name](batch)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(9216, 128)
        # self.fc2 = nn.Linear(128, 10)
        self.fc3 = nn.Linear(128, 10)
        self.cip = ConvolutionalSelfAttention(
            spatial_shape=[26, 26, 32],
            filter_size=3,
            approach_args={
                'name': '3',
                'pos_emb_dim': 10,
                'softmax_temp': 1
            },
            )

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        # pdb.set_trace()
        x = self.cip(x.permute(0, 2, 3, 1))
        x = x.permute(0, 3, 1, 2)
        x = F.max_pool2d(x, 12)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # pdb.set_trace()
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    import os; os.makedirs('/srv/share4/gstoica/data', exist_ok=True)
    dataset1 = datasets.MNIST('/srv/share4/gstoica/data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('/srv/share4/gstoica/data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "/srv/share4/gstoica/data/mnist_cnn.pt")


if __name__ == '__main__':
    main()