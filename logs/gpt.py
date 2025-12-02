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

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            γ1, γ2, γ3 = group["hyper_lr"]
            eps = group["eps"]

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

                    β1, β2 = group["betas"]
                    state["beta1_t"] = β1
                    state["beta2_t"] = β2

                    state["prod_beta1"] = β1
                    state["prod_beta2"] = β2

                state["step"] += 1

                m = state["m"]
                v = state["v"]

                β1_t = state["beta1_t"]
                β2_t = state["beta2_t"]

                prod_beta1 = state["prod_beta1"]
                prod_beta2 = state["prod_beta2"]

                if group["weight_decay"] != 0:
                    grad = grad.add(group["weight_decay"], p.data)

                # -------------------------------
                # ОБНОВЛЕНИЕ НЕВЗВЕШЕННЫХ МОМЕНТОВ
                # -------------------------------
                m_prev = m.clone()
                v_prev = v.clone()

                m.mul_(β1_t).add_(grad, alpha=1 - β1_t)
                v.mul_(β2_t).addcmul_(grad, grad, value=1 - β2_t)

                # -------------------------------
                # ОБНОВЛЕНИЕ ПРОИЗВЕДЕНИЙ БЕТ
                # -------------------------------
                prod_beta1 *= β1_t
                prod_beta2 *= β2_t

                state["prod_beta1"] = prod_beta1
                state["prod_beta2"] = prod_beta2

                # -------------------------------
                # bias-corrected моменты
                # -------------------------------
                m_hat = m / (1 - prod_beta1)
                v_hat = v / (1 - prod_beta2)

                # -------------------------------
                # ОБНОВЛЕНИЕ АЛФЫ (шага)
                # α_t = α_{t-1} + γ1 g_t m_hat / sqrt(v_hat+eps)
                # -------------------------------
                update_direction = m_hat / (v_hat.sqrt() + eps)
                group["lr"] += γ1 * torch.dot(grad.view(-1), update_direction.view(-1))

                α_t = group["lr"]

                # -------------------------------
                # ОБНОВЛЕНИЕ β1_t
                # eq:
                # β1_t = β1_{t-1} + γ2 * α_t * g_t * d/dβ1(...)
                # -------------------------------

                if state["step"] > 2:
                    # произведение для предыдущего шага
                    prod_beta1_prev = prod_beta1 / β1_t

                    d_bias = (
                        (1 - prod_beta1_prev) * (m_prev - grad) +
                        m * (prod_beta1_prev / β1_t)
                    ) / (1 - prod_beta1_prev) ** 2

                    delta_beta1 = γ2 * α_t * torch.dot(grad.view(-1),
                                                       (d_bias / (v_hat.sqrt() + eps)).view(-1))

                    β1_t = β1_t + delta_beta1
                    β1_t = float(torch.clamp(torch.tensor(β1_t), 0.8, 0.9999))

                # -------------------------------
                # ОБНОВЛЕНИЕ β2_t
                # eq:
                # β2_t = β2_{t-1} − γ3 α_t g_t m_hat/(2(v_hat+eps)^{3/2}) * ...
                # -------------------------------
                if state["step"] > 2:
                    prod_beta2_prev = prod_beta2 / β2_t

                    d_bias2 = (
                        (1 - prod_beta2_prev) * (v_prev - grad * grad) +
                        v * (prod_beta2_prev / β2_t)
                    ) / (1 - prod_beta2_prev) ** 2

                    delta_beta2 = γ3 * α_t * torch.dot(
                        grad.view(-1),
                        (m_hat / (2 * (v_hat + eps) ** 1.5) * d_bias2).view(-1)
                    )

                    β2_t = β2_t - delta_beta2
                    β2_t = float(torch.clamp(torch.tensor(β2_t), 0.9, 0.99999))

                # сохранить бетты
                state["beta1_t"] = β1_t
                state["beta2_t"] = β2_t

                # -------------------------------
                # ОБНОВЛЕНИЕ ПАРАМЕТРОВ
                # θ_t = θ_{t-1} − α_t * m_hat / sqrt(v_hat + eps)
                # -------------------------------
                p.data.add_(update_direction, alpha=-α_t)

        return loss
