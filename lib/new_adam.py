import math
import torch
from torch.optim import Optimizer


class AdamDynamic(Optimizer):
    def __init__(self, params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 hyper_lr=(1e-8, 1e-8, 1e-8)):
        defaults = dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay,hyper_lr=hyper_lr)
        super().__init__(params, defaults)
        self.writer = None

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            # best results: (0.0, 1e-0, 0.0)
            gamma_1, gamma_2, gamma_3 = group["hyper_lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Sparse grads not supported")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0

                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                    beta1_t, beta2_t = group["betas"]
                    state["beta1_t"] = beta1_t
                    state["beta2_t"] = beta2_t
                    state["prod_beta1"] = beta1_t
                    state["prod_beta2"] = beta2_t
                    state["g_prev"] = torch.zeros_like(p.data)

                state["step"] += 1

                m = state["m"]
                v = state["v"]

                beta1_t = state["beta1_t"]
                beta2_t = state["beta2_t"]

                prod_beta1 = state["prod_beta1"]
                prod_beta2 = state["prod_beta2"]

                if group["weight_decay"] != 0:
                    grad = grad.add(group["weight_decay"], p.data)

                if state["step"] > 1:
                    # part of alpha calculations ...
                    prev_bias_correction1 = 1 - prod_beta1
                    prev_bias_correction2 = 1 - prod_beta2
                    h = torch.dot(grad.view(-1),torch.div(m,(v.sqrt()+group['eps'])).view(-1))*math.sqrt(prev_bias_correction2)/prev_bias_correction1
                    
                    if state["step"] > 2:
                        # part of beta1_t calculations ...
                        prod_beta1_prev = prod_beta1 / beta1_t
                        prod = 1/((v/prev_bias_correction1).sqrt()+group["eps"])
                        d_bias = ((1/beta1_t)*(m-state["g_prev"])+(state["g_prev"]*prod_beta1_prev))/((1-(prod_beta1*beta1_t))**2)
                        delta_beta1 = gamma_2*group["lr"]*torch.dot(grad.view(-1),(d_bias*prod).view(-1))
                        state["beta1_t"] += delta_beta1
                        state["beta1_t"] = float(torch.clamp(state["beta1_t"], 1e-5, 1.0-1e-5))
                        
                        # part of beta2_t calculations ...
                        prod_beta2_prev = prod_beta2 / beta2_t
                        prod = (m/prev_bias_correction2)/(2*((v/prev_bias_correction2).sqrt()+group["eps"]))/(((v/prev_bias_correction2).sqrt()+group["eps"]).pow(2))
                        d_bias2 = ((1/beta2_t)*(v-state["g_prev"]*state["g_prev"])+(state["g_prev"]*state["g_prev"]*prod_beta2_prev))/((1-(prod_beta2*beta2_t))**2)
                        delta_beta2 = -gamma_3*group["lr"]*torch.dot(grad.view(-1),(d_bias2*prod).view(-1))
                        state["beta2_t"] += delta_beta2
                        state["beta2_t"] = float(torch.clamp(state["beta2_t"], 1e-5, 1.0-1e-5))
                    group['lr'] += gamma_1 * h
                    group['lr'] = float(torch.clamp(group['lr'], 1e-10, 1e-2))


                state["g_prev"] = grad.clone()

                state["prod_beta1"] = prod_beta1 * state['beta1_t']
                state["prod_beta2"] = prod_beta2 * state['beta2_t']
                beta1_t = state["beta1_t"]
                beta2_t = state["beta2_t"]
                prod_beta1 = state["prod_beta1"]
                prod_beta2 = state["prod_beta2"]

                m.mul_(beta1_t).add_(1 - beta1_t, grad)
                v.mul_(beta2_t).addcmul_(1 - beta2_t, grad, grad)
                denom = v.sqrt().add_(group["eps"])
                step_size = group["lr"] * math.sqrt(1 - prod_beta2) / (1 - prod_beta1)
                p.data.addcdiv_(-step_size, m, denom)

        if self.writer is not None:
            self.writer.add_scalar("Train/alpha",   group['lr'], state['step'])
            self.writer.add_scalar("Train/beta1_t", state['beta1_t'], state['step'])
            self.writer.add_scalar("Train/beta2_t", state['beta2_t'], state['step'])
        return loss