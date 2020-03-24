import torch
import torch.nn as nn
import torch.distributions as D
from torch import optim
import torch.nn.functional as F
import numpy as np

class PolicyPretrainer():

    def __init__(self, model, hparams, device):
        self.load_path = hparams.load_path
        self.model = model
        self.device = device
        self.loss_criterion = nn.BCEWithLogitsLoss()

    def _setup(self, env):
        self.env = env
        self.losses = []

    def _compute_loss(self):
        self.losses.append(self.loss.item())

    def train(self, env, optimizer, n_iters, idx):
        self.is_eval = False
        self._setup(env)
        self.model.train()

        for iter in range(1, n_iters + 1):

            optimizer.zero_grad()

            # numpy shape (batch_size, 8, 3), float32
            # numpy shape (batch_size, 8, 2048), float32
            # numpy shape (batch_size, ), float32
            # list length batch_size
            a_t, f_t, swapped_target, _ = self.env.generate_next_minibatch()

            # convert to tensors
            a_t = torch.from_numpy(a_t).to(self.device)
            f_t = torch.from_numpy(f_t).to(self.device)
            swapped_target = torch.from_numpy(swapped_target).to(self.device)

            # model make prediction
            # shape (batch_size,)
            logit = self.model(f_t, a_t)

            # make self.loss
            self.loss = self.loss_criterion(logit, swapped_target)

            # backprop
            self.loss.backward()
            optimizer.step()

            # log training loss
            self._compute_loss()

    def test(self, env, idx):
        self.is_eval = True
        self._setup(env)
        self.model.eval()

        self.results = {}
        looped = False

        with torch.no_grad():
            # Loop through batches until all datapts are covered
            while True:
                
                a_t, f_t, swapped_target, instr_ids = self.env.generate_next_minibatch()
                a_t = torch.from_numpy(a_t).to(self.device)
                f_t = torch.from_numpy(f_t).to(self.device)
                swapped_target_t = torch.from_numpy(swapped_target).to(self.device)
                logit = self.model(f_t, a_t)
                self.loss = self.loss_criterion(logit, swapped_target_t)
                self._compute_loss()

                for i, instr_id in enumerate(instr_ids):
                    if instr_id in self.results:
                        looped = True
                    else:
                        self.results[instr_id] = {
                            'logit': logit[i].item(),
                            'target': swapped_target[i]
                        }
                if looped:
                    break

    def write_results(self):
        output = []
        for k, v in self.results.items():
            item = { 'instr_id' : k }
            item.update(v)
            output.append(item)
        
        with open(self.results_path, 'w') as f:
            try:
                json.dump(output, f)
            except:
                import ipdb; ipdb.set_trace()
