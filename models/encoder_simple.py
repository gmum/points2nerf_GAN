import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['E']['use_bias']
        self.relu_slope = config['model']['E']['relu_slope']

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=64, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, (512 + self.z_size) // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear((512 + self.z_size) // 2, self.z_size, bias=True),
        )

    def forward(self, x):
        output = self.conv(x)
        output2 = output.max(dim=2)[0]
        z = self.fc(output2)
        return z

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False


class Encoder_lrelu(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['E']['use_bias']
        self.relu_slope = config['model']['E']['relu_slope']

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=64, kernel_size=1, bias=self.use_bias),
            nn.LeakyReLU(negative_slope=self.relu_slope, inplace=True),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, bias=self.use_bias),
            nn.LeakyReLU(negative_slope=self.relu_slope, inplace=True),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, bias=self.use_bias),
            nn.LeakyReLU(negative_slope=self.relu_slope, inplace=True),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, bias=self.use_bias),
            nn.LeakyReLU(negative_slope=self.relu_slope, inplace=True),

            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, bias=self.use_bias),
            nn.LeakyReLU(negative_slope=self.relu_slope, inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, (512 + self.z_size) // 2, bias=True),
            nn.LeakyReLU(negative_slope=self.relu_slope, inplace=True),
            nn.Linear((512 + self.z_size) // 2, self.z_size, bias=True),
        )

    def forward(self, x):
        output = self.conv(x)
        output2 = output.max(dim=2)[0]
        z = self.fc(output2)
        return z

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
