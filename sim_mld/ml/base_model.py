import os.path

import random
from collections import deque
import numpy as np
import torch
from torch import optim
from torch_geometric.data import Dataset, Data, Batch

from sim_src.util import USE_CUDA, STATS_OBJECT, hard_update_inplace, soft_update_inplace


class base_model(STATS_OBJECT):
    def __init__(self,  LR = 0.0001, TAU = 0.001, WITH_TARGET = False):
        self.WITH_TARGET = WITH_TARGET
        self.LR = LR
        self.TAU = TAU
        self.model = None
        self.model_target = None
        self.model_optim = None

        self.init_model()
        
        self.init_optim()
        
        self.move_model_to_gpu()
    
    def move_model_to_gpu(self):
        if USE_CUDA:
            self.model.to(torch.cuda.current_device())
            if self.WITH_TARGET:
                self.model_target.to(torch.cuda.current_device())
            print(self.__class__.__name__,"is on gpu")
        else:
            print(self.__class__.__name__,"is on cpu")

    def init_model(self):
        pass

    def init_optim(self):
        self.model_optim = optim.Adam(self.model.parameters(), lr=self.LR)

    def load_model(self, path, path_target = None):
        if not path_target:
            path_target = path
        self.model = self._load(path)
        if self.WITH_TARGET:
            self.model_target = self._load(path_target)
        self.init_optim()
        
    def _load(self, path):
        if USE_CUDA:
            return torch.load(path, map_location=torch.device('cuda', torch.cuda.current_device()), weights_only=False)
        else:
            return torch.load(path, map_location=torch.device('cpu'),weights_only=False)
    
    def update_target_nn(self,hard=False):
        if self.WITH_TARGET:
            if hard:
                soft_update_inplace(self.model_target, self.model, self.TAU)
            else:
                hard_update_inplace(self.model_target, self.model)
            
    def save(self, path: str, postfix: str):
        try:
            os.mkdir(path)
        except:
            pass
        torch.save(self.model, os.path.join(path, self.__class__.__name__+ "." + postfix + ".pt"))
        if self.WITH_TARGET:
            torch.save(self.model_target, os.path.join(path, self.__class__.__name__+ "_target." + postfix + ".pt"))

    def step(self, batch):
        pass

    def get_output_np(self, input_np:np.ndarray)->np.ndarray:
        pass
    
    def eval(self):
        if self.model is not None:
            self.model.eval()
        if self.model_target is not None:
            self.model_target.eval()
            
            
            
class ReplayMemory(Dataset):
    def __init__(self, capacity: int):
        """
        Fixed-size replay buffer for PyG Data objects.
        """
        self.capacity = capacity
        # deque with maxlen gives automatic FIFO drops
        self.buffer = deque(maxlen=capacity)
    
    def push(self, data: Data):
        """
        Add a new graph Data object to the buffer.
        If full, the oldest graph is discarded.
        """
        # Optionally clone or deepcopy if data reused externally
        self.buffer.append(data)
    
    def __len__(self) -> int:
        # Current number of stored graphs
        return len(self.buffer)
    
    def __getitem__(self, idx: int) -> Data:
        # Indexing into the deque is O(1) in CPython
        return self.buffer[idx]
    
    def sample(self, batch_size: int) -> list[Data]:
        """
        Utility to sample a raw list of Data graphs without DataLoader.
        """
        if batch_size > len(self.buffer):
            return None
        ret = random.sample(self.buffer, batch_size)
        # clear the buffer
        self.buffer.clear()
        return ret