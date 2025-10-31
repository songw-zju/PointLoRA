"""
Implementation of LoRA (LoRA: Low-Rank Adaptation of Large Language Models: https://arxiv.org/abs/2106.09685)
Codes are modified from (https://github.com/microsoft/LoRA/blob/main/loralib/layers.py)
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from functools import reduce
from operator import mul


class LoRALayer(nn.Module):
    """
    Base lora class
    """
    def __init__(
            self,
            r,
            lora_alpha,
         ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        # Mark the weight as unmerged
        self.merged = False

    def reset_parameters(self):
        raise NotImplementedError

    def train(self, mode:bool = True):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError


class LoRALinear(LoRALayer):
    def __init__(self, r, lora_alpha, linear_layer):
        """
        LoRA class for nn.Linear class
        :param r: low rank dimension
        :param lora_alpha: scaling factor
        :param linear_layer: target nn.Linear layer for applying Lora
        """
        super().__init__(r, lora_alpha)
        self.linear = linear_layer

        in_features = self.linear.in_features
        out_features = self.linear.out_features

        # Lora configuration
        self.lora_A = nn.Parameter(self.linear.weight.new_zeros((r, in_features)))
        self.lora_B = nn.Parameter(self.linear.weight.new_zeros((out_features, r)))
        self.scaling = self.lora_alpha / self.r
        self.weight = nn.Parameter(torch.tensor([1.0]))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)


    def train(self, mode:bool = True):
        self.linear.train(mode)
        # if self.merged:
        #     self.linear.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
        #     self.merged = False


    def eval(self):
        self.linear.eval()
        # if not self.merged:
        #     self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        #     self.merged = True


    def forward(self, x, prompt_mlp):
        # set self.merged to False for training
        self.merged = False
        if not self.merged:
            result = F.linear(x, self.linear.weight, bias=self.linear.bias)
            out = (x @ self.lora_A.T @ self.lora_B.T)
            h_x = prompt_mlp(x)
            out = out + self.weight * h_x
            result += out
            return result
        else:
            return F.linear(x, self.linear.weight, bias=self.linear.bias)


class LoraConv2d(LoRALayer):
    def __init__(self, r, lora_alpha, conv_layer):
        """
        LoRA class for nn.Conv2d class
        """
        super().__init__(r, lora_alpha)
        self.conv = conv_layer

        in_channels = self.conv.in_channels
        out_channels = self.conv.out_channels
        kernel_size = self.conv.kernel_size[0]

        # lora configuration
        self.lora_A = nn.Parameter(
            self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
        )
        self.lora_B = nn.Parameter(
            self.conv.weight.new_zeros((out_channels * kernel_size, r * kernel_size))
        )
        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        self.conv.train(mode)
        if self.merged:
            # Make sure that the weights are not merged
            self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
            self.merged = False

    def eval(self):
        self.conv.eval()
        if not self.merged:
            # Merge the weights and mark it
            self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x):
        if not self.merged:
            return F.conv2d(
                x,
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups
            )
        return self.conv(x)


