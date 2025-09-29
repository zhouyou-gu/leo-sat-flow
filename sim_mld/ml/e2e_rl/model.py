import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR


from sim_src.util import *

from sim_mld.ml.base_model import base_model, ReplayMemory
from sim_mld.ml.e2e_rl.nn import AC_NN 

from torch_geometric.data import Data, Batch

torch.autograd.set_detect_anomaly(True)

from torch_geometric.data import Data

import plotext

class rl_model(base_model):
    def __init__(self, LR =0.001, TAU = 0.001, BETA=0.5, GAMMA=0.001, DET=True):
        self.BETA = BETA
        self.GAMMA = GAMMA
        base_model.__init__(self, LR = LR, TAU=TAU, WITH_TARGET = True)
        self.batch_size = 1
        self.GAMMA = GAMMA/self.batch_size
        self.data_set = ReplayMemory(self.batch_size)
        self.DET = DET
        self.mean_rwd = 0.

        
    def init_model(self):
        self.model = AC_NN()
        self.model_target = AC_NN()
        self.update_target_nn(hard=True)

    def init_optim(self):
        self.actor_optim = torch.optim.Adam(self.model.parameters(), lr=self.LR)
        self.lr_scheduler_actor = LambdaLR(self.actor_optim, lr_lambda=lambda epoch: 1.0/(epoch+1)**self.BETA)

        self.critic_optim = torch.optim.Adam(self.model.parameters(), lr=self.LR)
        self.lr_scheduler_critic = LambdaLR(self.critic_optim, lr_lambda=lambda epoch: 1.0/(epoch+1)**self.BETA)

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
            st_pair_index=to_tensor(data["st_pair_index"],dtype=LONG_TYPE).T,
            st_pair_attr=to_tensor(data["st_pair_attr"]),
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
        if self.DET:
            q_approx = self.model.evaluate(batch.x, batch.cp_edge_index, batch.cp_edge_attr, batch.pr_edge_attr, batch.st_pair_index)
            q_approx = q_approx.squeeze()
            print("Q approx shape:", q_approx.shape, "ST pair attr shape:", batch.st_pair_attr.shape, "st_pair_index shape:", batch.st_pair_index.shape)
            loss_eva = torch.nn.functional.mse_loss(q_approx, batch.st_pair_attr, reduction="mean")
            self.critic_optim.zero_grad()
            loss_eva.backward()
            self.critic_optim.step()
            self.critic_optim.zero_grad()
            
            prices = self.model.forward(batch.x,batch.cp_edge_index,batch.cp_edge_attr).squeeze()
            q_approx = self.model.evaluate(batch.x, batch.cp_edge_index, batch.cp_edge_attr, prices, batch.st_pair_index)
            q_approx = q_approx.squeeze()
            loss_act = -q_approx.mean()
        else:
            prices = self.model.forward(batch.x,batch.cp_edge_index,batch.cp_edge_attr).squeeze()
            prices = torch.clamp(prices,min=1e-5)
            self.mean_rwd = 0.1*batch.st_pair_attr.mean().item() + self.mean_rwd*0.9
            
            prices = torch.clamp(prices, min=1e-5, max=1-1e-5)
            prices_logit = torch.log(prices/(1-prices))
            std_of_normal = 1.0
            
            prices_sample = batch.pr_edge_attr
            prices_sample = torch.clamp(prices_sample, min=1e-5, max=1-1e-5)
            prices_sample_logit = torch.log(prices_sample/(1-prices_sample))
            log_prob = - ((prices_logit - prices_sample_logit)**2) / (2*std_of_normal**2) - torch.log(prices*(1-prices)*std_of_normal * (2*math.pi)**0.5)
            log_prob = log_prob.mean()
            
            loss_act = - log_prob * (batch.st_pair_attr.mean().item()-self.mean_rwd)
            
        self.actor_optim.zero_grad()
        loss_act.backward()
        self.actor_optim.step()
        self.actor_optim.zero_grad()
        
        subg = (batch.qx_edge_attr - batch.rc_edge_attr)
        print("subg", subg)
        print("counting subg>0", (subg > 0).sum().item())
        print("counting subg==0", (subg == 0).sum().item())
        print("counting subg<0", (subg < 0).sum().item())
        print("qx_edge_attr", batch.qx_edge_attr)
        print("counting qx>0", (batch.qx_edge_attr > 0).sum().item())
        print("counting qx<=0", (batch.qx_edge_attr <= 0).sum().item())
        print("rc_edge_attr", batch.rc_edge_attr)
        print("counting rc>0", (batch.rc_edge_attr > 0).sum().item())
        print("counting rc<=0", (batch.rc_edge_attr <= 0).sum().item())
        try:
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
        except Exception as e:
            print("Plotext error:", e)
            pass
        
        self.lr_scheduler_actor.step()
        self.lr_scheduler_critic.step()
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
            
        if self.DET:
            pass
        else:
            # sample from logistic normal distribution
            # inverse of sigmoid
            prices = torch.clamp(prices, min=1e-5, max=1-1e-5)
            mean_of_normal = torch.log(prices / (1 - prices))
            std_of_normal = 1.0
            normal_sample = torch.normal(mean_of_normal, std_of_normal)
            prices = torch.sigmoid(normal_sample)
        return to_numpy(prices)

    
    
    