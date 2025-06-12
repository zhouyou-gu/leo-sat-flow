import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR


from sim_src.util import *

from sim_mld.ml.base_model import base_model, ReplayMemory
from sim_mld.ml.ld_sgl.nn import PriceGNN 

from torch_geometric.data import Data, Batch

torch.autograd.set_detect_anomaly(True)

from torch_geometric.data import Data

import plotext
class ld_model(base_model):
    def __init__(self, LR =0.001, TAU = 0.001):
        base_model.__init__(self, LR = LR, TAU=TAU, WITH_TARGET = True)
        self.batch_size = 1
        self.data_set = ReplayMemory(self.batch_size)
        
    def init_model(self):
        self.model = PriceGNN()
        self.model_target = PriceGNN()
        self.update_target_nn(hard=True)
        # if hasattr(torch, 'compile'):
        #     self.model = torch.compile(self.model)

    def init_optim(self):
        self.model_optim = torch.optim.Adam(self.model.parameters(), lr=self.LR, weight_decay=1e-5)
        # self.model_optim = SGD(self.model.parameters(), lr=self.LR, weight_decay=1e-5)
        self.lr_scheduler = LambdaLR(self.model_optim, lr_lambda=lambda epoch: 1.0/(epoch+1)**0.2)

    def _add_graph(self, data):
        """
        Add a new graph to the dataset, assuming the setup is
        data["x"] = self.positions
        data["cp_edge_index"] = cp_edge_list[:, 0:2]
        data["cp_edge_attr"] = cp_edge_list[:, 2]
        data["qx_edge_index"] = qx_edge_list[:, 0:2]
        data["qx_edge_attr"] = qx_edge_list[:, 2]
        data["rc_edge_index"] = rc_edge_list[:, 0:2]
        data["rc_edge_attr"] = rc_edge_list[:, 2]
        """
        data_tensor = Data(
            x=to_tensor(data["x"]),
            cp_edge_index=to_tensor(data["cp_edge_index"],dtype=LONG_TYPE).T,
            cp_edge_attr=to_tensor(data["cp_edge_attr"]),
            qx_edge_index=to_tensor(data["qx_edge_index"],dtype=LONG_TYPE).T,
            qx_edge_attr=to_tensor(data["qx_edge_attr"]),
            rc_edge_index=to_tensor(data["rc_edge_index"],dtype=LONG_TYPE).T,
            rc_edge_attr=to_tensor(data["rc_edge_attr"]),
            pr_edge_index=to_tensor(data["pr_edge_index"],dtype=LONG_TYPE).T,
            pr_edge_attr=to_tensor(data["pr_edge_attr"]),
        )
        self.data_set.push(data_tensor)
    
    def _get_batch(self):
        if len(self.data_set) == 0:
            return None
        data_list = self.data_set.sample(self.batch_size)
        if not data_list:
            return None
        batch = Batch.from_data_list(data_list)
        return batch
    
    @counted 
    def step(self, data):        
        if not data:
            print("None batch in step", self.N_STEP)
            return
        
        self._add_graph(data)
        
        batch = self._get_batch()
        if not batch:
            print("None batch in step", self.N_STEP)
            return
        else:
            print("training with batch", self.N_STEP, batch.size)
        
        max_vals = []
        for key, tensor in self.model.state_dict().items():
            if 'weight' in key:
                max_vals.append(tensor.abs().max().item())
        print("Max weight:", max(max_vals))
        prices = self.model.forward(batch.x, batch.cp_edge_index, batch.cp_edge_attr)   
        subg = (batch.qx_edge_attr - batch.rc_edge_attr)
        print("subg", subg)
        print("qx_edge_attr", batch.qx_edge_attr)
        print("rc_edge_attr", batch.rc_edge_attr)
        print("prices", prices)

        plotext.title("Prices Distribution")
        plotext.hist(np.log10(to_numpy(prices)+1e-5), bins=50, norm=True)
        plotext.plotsize(100, 30)
        plotext.show()
        plotext.clf()
        plotext.title("SubG Distribution")
        plotext.hist(to_numpy(subg), bins=50, norm=True)
        plotext.plotsize(100, 30)
        plotext.show()
        plotext.clf()
        plotext.title("Qx Edge Attr Distribution")
        plotext.hist(np.log10(to_numpy(batch.qx_edge_attr)+1e-5), bins=50, norm=True)
        plotext.plotsize(100, 30)
        plotext.show()
        plotext.clf()
        plotext.title("RC Edge Attr Distribution")
        plotext.hist(np.log10(to_numpy(batch.rc_edge_attr)+1e-5), bins=50, norm=True)
        plotext.plotsize(100, 30)
        plotext.show()
        plotext.clf()
        loss_d = -prices * subg

        loss_d_mean = torch.mean(loss_d)
        self._add_np_log("loss_d",self.N_STEP,[loss_d_mean.item()])

        loss = loss_d_mean
        self._add_np_log("loss_all",self.N_STEP,[loss.item()])

        text = f"loss: {loss.item():.4f}, loss_d: {loss_d_mean.item():.4f}"
        self._printalltime(text)
        self.model_optim.zero_grad()
        loss.backward()
        self.model_optim.step()
        self.model_optim.zero_grad()
        self.lr_scheduler.step()
        self.update_target_nn(hard=False)
        self.clear_memory()



    @torch.no_grad()
    def get_output_np_edge_weight(self, x, cp_edge_index, cp_edge_attr, use_target=True):
        
        x = to_tensor(x)
        cp_edge_index = to_tensor(cp_edge_index, dtype=LONG_TYPE).T
        cp_edge_attr = to_tensor(cp_edge_attr)
        if use_target:
            prices = self.model_target.forward(x, cp_edge_index, cp_edge_attr)
        else:
            prices = self.model.forward(x, cp_edge_index, cp_edge_attr)
        return to_numpy(prices)

    
    
    