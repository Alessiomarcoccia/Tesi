import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial


#temporal feature aggregation module OR Local Temporal Excitation(LTE)
class TFAM(nn.Module):
    def __init__(self,in_channels):
        super(TFAM,self).__init__()

        #LTE
        reduced_channels = in_channels // self.reduction
        self.rc = reduced_channels
        self.ic = in_channels
        #apply global spatial averaging pooling of the input tensor in the channel direction
        #input shape [N, C, T, H, W], output shape [N,C,T,1,1]
        self.global_avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        # First 1D temporal convolution (W3) with kernel size 3 and reduced channels
        self.conv_reduce = nn.Conv1d(in_channels=in_channels, out_channels=reduced_channels, kernel_size=3,padding=1)

        # Batch normalization and ReLU for the first convolution
        self.bn_relu = nn.Sequential(
            nn.BatchNorm1d(reduced_channels),
            nn.ReLU()
        )

        # Second 1D temporal convolution (W4) with kernel size 1 to restore original channel dimension
        self.conv_process = nn.Conv1d(in_channels=reduced_channels, out_channels=in_channels, kernel_size=1)

        # Sigmoid activation for attention mask
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # Input shape: [N,T,C,H,W]

        #Permute input shape to [N,C,T,H,W]
        x = x.permute(0,2,1,3,4)

        out = self.global_avg_pool(x) # Resulting shape: (N, C, T, 1, 1)

        out = out.squeeze(-1).squeeze(-1) # Resulting shape: (N, C, T)

        out = self.conv_reduce(out) # Output shape: (N, C/r, T)

        out = self.bn_relu(out) # Output shape: (N, C/r, T)

        out = self.conv_process(out) # Output shape: (N, C, T)

        out = self.sigmoid(out) # Output shape: (N, C, T)

        out = out.unsqueeze(-1).unsqueeze(-1) # Output shape: (N, C, T, 1, 1)

        #element-wise multiplication
        out = out * x + x  # Output shape: (N, C, T, H, W)

        out = out.permute(0,2,1,3,4) # Output shape: (N, T, C, H, W)


        return out

#local grouping convolution superposition module
class LGCSM(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(LGCSM,self).__init__()
        self.split_channels = out_channels // 4

        # 3x3 convolutions for second, third, and fourth segments
        self.conv1 = nn.Conv2d(self.split_channels, self.split_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(self.split_channels, self.split_channels, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(self.split_channels, self.split_channels, kernel_size=3, padding=1, bias=False)


        # Batch normalization and ReLU activation
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        # Split input into 4 segments along the channel dimension
        N, T, C_in, H, W = x.shape

        # Split input into 4 segments along the channel dimension
        X1, X2, X3, X4 = x.chunk(4, dim=2)

        # Apply convolution to the first segment (no skip connection for X1)
        X1_out = X1

        # Apply convolution to the second segment
        X2_4D = X2.mean(dim=1)# N,C,H,W
        X2_out4D = self.conv1(X2_4D)
        X2_out5D = X2_out4D.unsqueeze(1).expand(N, T, -1, H, W)
        #X2_out5D = X2_out4D.reshape(N, T, C_in // 4, H, W)

        # Apply convolution to the third segment with skip connection from X2
        X3_out4D = X2.mean(dim=1) # N,C,H,W
        X3_out4D = self.conv2(X3_out4D + X2_out4D)
        X3_out5D = X3_out4D.unsqueeze(1).expand(N, T, -1, H, W)
        #X3_out5D = X3_out4D.reshape(N, T, C_in // 4,H, W)

        # Apply convolution to the fourth segment with skip connection from X3
        X4_out4D = X3.mean(dim=1) # N,C,H,W
        X4_out4D = self.conv3(X4_out4D + X3_out4D)
        X4_out5D = X4_out4D.unsqueeze(1).expand(N, T, -1, H, W)
        #X4_out5D = X4_out4D.reshape(N, T, C_in // 4, H, W)

        # Concatenate the output from all segments
        output = torch.cat([X1_out, X2_out5D, X3_out5D, X4_out5D], dim=2)  # Concatenate along channel axis

        # Apply batch normalization and ReLU activation
        output = output.permute(0, 2, 1, 3, 4)  # Permute to [N, C, T, H, W]
        output = self.bn(output)
        output = output.permute(0, 2, 1, 3, 4)  # Permute back to [N, T, C, H, W]
        output = self.relu(output)

        return output


class STFEBlock(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super(STFEBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.TFAM = TFAM(out_channels)
        self.LGCSM = LGCSM(out_channels,out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=stride, bias=False)


    def forward(self, x):
        #ricorda che devi manipolare x,infatti esso verrà dato in input con 5 dimensioni,ma per la prima convoluzione ne deve avere solo 4,
        #poi deve tornare ad averne 5
        residual = x
        N, T, C, H, W = x.shape

       # print("Shape of x:", x.shape)

        #x_reshaped = x.reshape(N*T, C, H, W)
        x_reshaped = x.mean(dim=1) # -> N,C,H,W
        out = self.conv1(x_reshaped) #output [N, out_channels, H, W]
        out = out.unsqueeze(1) #output [N, 1, out_channels, H, W]
        out = out.expand(N, T, -1, H, W) #output [N, T, out_channels, H, W]
        #out_reshaped = out.reshape(N, T, -1, H, W) #output [N, T, out_channels, H, W]

        out = self.TFAM(out) #output [N, T, out_channels, H, W]
        out = self.LGCSM(out) #output [N, T, out_channels, H, W]
        #print(f"Shape dopo LGCSM{out.shape}")

        out = out.mean(dim=1) #output [N, out_channels, H, W]
        out = self.conv2(out) #output [N, out_channels, H, W]
        out = out.unsqueeze(1) #output [N, 1, out_channels, H, W]
        out = out.expand(N, T, -1, H, W) #output [N, T, out_channels, H, W]

        out = out.clone() + residual

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, last_fc=True):
        super(ResNet, self).__init__()
        self.last_fc = last_fc
        self.in_channels = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2,padding=3, bias=False)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 32, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 32, layers[3], stride=1)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=0.5)

        self.fc = nn.Linear(32, num_classes)


    def _make_layer(self, block, out_channels, blocks, stride=1):

        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        #self.in_channels = out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        #print("Shape of x:", x.shape)
        N, T, C, H, W = x.shape
        x_reshaped = x.mean(dim=1)# -> shape: N,C,H,W

        x = self.conv1(x_reshaped)

        x = x.unsqueeze(1)  # Add a singleton T dimension -> [N, 1, 64, H, W]
        x = x.expand(N, T, -1, 112, 112)  # Shape becomes [N, T, 64, H, W]


        x = self.layer1(x)
        #print("finito layer1")
        x = self.layer2(x)
        #print("finito layer2")
        x = self.layer3(x)
        #print("finito layer3")
        x = self.layer4(x)
        #print("finito layer4")

        x = x.permute(0, 2, 1, 3, 4)

        x = self.avgpool(x)
        #print(f"X shape after avpool:{x.shape}")
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        #print("Shape of x before fc(flattened):", x.shape)
        if self.last_fc:
            x = self.fc(x)
        #print("finito")
        return x


def resnet50():
    """Constructs a ResNet-50 model.
    """
    model = ResNet(STFEBlock, [3, 4, 6, 3])
    return model

