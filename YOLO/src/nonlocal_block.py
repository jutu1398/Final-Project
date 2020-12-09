import torch.nn as nn
import torch
import torch.nn.functional as F

class gaussian_nonlocal(nn.Module):
    def __init__(self, in_channels):
        super(gaussian_nonlocal, self).__init__()

        self.in_channels = in_channels
        self.block_channels = in_channels // 2
            
        self.g = nn.Conv2d(in_channels = self.in_channels, out_channels = self.block_channels, kernel_size = 1)
        
        
        self.Wz = nn.Sequential(
             nn.Conv2d(in_channels=self.block_channels, out_channels=self.in_channels, kernel_size=1),
             nn.BatchNorm2d(self.in_channels)
         )
    
    def forward(self, x):
        #x of size N, C, H, W
        
        n, c, h, w = x.size()
        
        identity = x

        g = self.g(x)
        
        #gaussian does not use 1x1 conv, change to 
        theta = x.view(n, self.in_channels,-1)
        phi = x.view(n, self.in_channels, -1)
        theta = theta.permute(0,2,1)
       
        f = torch.matmul(theta, phi)
        f = F.softmax(f, 1)
  
        g = g.view(n, self.block_channels, -1)
        g = g.permute(0, 2, 1)
     
        
        y = torch.matmul(f, g)
        y = y.transpose(1,2)
        y = y.view(n, self.block_channels, h, w)
        
        y = self.Wz(y)
        
        z = x + y
        return y