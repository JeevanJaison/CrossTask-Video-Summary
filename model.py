from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

# import torch.nn as nn

# class Model(nn.Module):
#     def __init__(self, d, M, A, q):
#         super(Model, self).__init__()
#         self.fc = nn.Linear(d,M)
#         self.m = nn.Dropout(p=q)
#         self.A = A
        
#     def forward(self, x, task):
#         return self.fc(self.m(x)).matmul(self.A[task])
    
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, d, M, A, q):
        super(Model, self).__init__()
        self.fc = nn.Linear(d, M)
        self.m = nn.Dropout(p=q)
        self.A = A  # Task-step mapping

    def forward(self, x, task):
        device = x.device  # Get the device of input `x`
        A_task = self.A[task].to(device)  # Move A[task] to the same device as x
        return self.fc(self.m(x)).matmul(A_task)
