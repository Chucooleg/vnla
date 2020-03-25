import torch
import torch.nn as nn
import torch.distributions as D
from torch import optim
import torch.nn.functional as F
import numpy as np
import json

class PolicyPretrainer():

    def __init__(self, model, hparams, device):
        self.load_path = hparams.load_path
        self.model = model
        self.device = device
        self.loss_criterion = nn.BCEWithLogitsLoss()
        self.coverage_size = hparams.coverage_size if hasattr(hparams, 'coverage_size') else None
        self.include_language = hparams.include_language

    def _setup(self, env):
        self.env = env
        self.losses = []

    def _compute_loss(self):
        self.losses.append(self.loss.item())

    def _propagate(self):

        # numpy shape (batch_size, 8, 3), float32
        # numpy shape (batch_size, 8, 2048), float32
        # numpy shape (batch_size, ), float32
        # list length batch_size
        a_t, f_t, swapped_target, instr_ids, seq, seq_mask, seq_lengths = self.env.generate_next_minibatch()        

        if self.include_language:
            # encoder encode language instructions
            ctx, _ = self.model.encode(seq, seq_lengths)

            # Coverage vector
            if self.coverage_size is not None:
                cov = torch.zeros(seq_mask.size(0), seq_mask.size(1), self.coverage_size, dtype=torch.float, device=self.device)
            else:
                cov = None       

        # convert to tensors
        a_t = torch.from_numpy(a_t).to(self.device)
        f_t = torch.from_numpy(f_t).to(self.device)
        swapped_target = torch.from_numpy(swapped_target).to(self.device)

        # model make swap prediction
        if self.include_language:
            logit, cov = self.model(f_t, a_t, ctx, seq_mask, cov)
        else:
            # shape (batch_size,)
            logit = self.model(f_t, a_t)            

        self.loss = self.loss_criterion(logit, swapped_target)
        return logit, swapped_target, instr_ids

    def train(self, env, optimizer, n_iters, idx):
        self.is_eval = False
        self._setup(env)
        self.model.train()

        for iter in range(1, n_iters + 1):

            optimizer.zero_grad()
            self._propagate()

            # backprop
            self.loss.backward()
            optimizer.step()

            # log training loss
            self._compute_loss()

    def test(self, env, idx):
        self.is_eval = True
        self._setup(env)
        self.model.eval()
        self.env.reset_epoch()

        self.results = {}
        looped = False

        with torch.no_grad():
            # Loop through batches until all datapts are covered
            while True:
            
                logit, swapped_target, instr_ids = self._propagate()
                self._compute_loss()

                for i, instr_id in enumerate(instr_ids):
                    if instr_id in self.results:
                        looped = True
                    else:
                        self.results[instr_id] = {
                            'logit': logit[i].item(),
                            'target': swapped_target[i].tolist()
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
            json.dump(output, f)

