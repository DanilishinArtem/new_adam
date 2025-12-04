import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import os

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    # Полная детерминированность cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dataset(n=5000):
    x = torch.linspace(-5, 5, n).unsqueeze(1)
    y = (
        0.4 * torch.sin(3 * x)
        + 0.3 * torch.sin(7 * x + 1)
        + 0.2 * torch.cos(5 * x - 2)
        + 0.1 * x**2
        + 0.05 * x**3
        + 0.2 * torch.randn_like(x)
    )
    return x, y


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(optimizer, model, time_to_learn, optimizer_adam: str = 'adam'):
    x, y = make_dataset()
    

    losses = []
    alphas = []
    betas1 = []
    betas2 = []
    for step in range(time_to_learn):
        optimizer.zero_grad()
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        p = list(model.parameters())[0]
        st = optimizer.state[p]
        alphas.append(optimizer.param_groups[0]["lr"])
        if optimizer_adam == 'new_adam':
            betas1.append(st["beta1_t"])
            betas2.append(st["beta2_t"])
        if step % 200 == 0:
            if optimizer_adam == 'adam':
                print(f"step={step:4d}, loss={loss.item():.6f}")
            elif optimizer_adam == 'new_adam':
                print(
                    f"step={step:4d}, loss={loss.item():.6f}, "
                    f"α={optimizer.param_groups[0]['lr']:.6f}, "
                    f"β1={st['beta1_t']:.6f}, β2={st['beta2_t']:.6f}"
                )
            else:
                print(
                    f"step={step:4d}, loss={loss.item():.6f}, "
                    f"α={optimizer.param_groups[0]['lr']:.6f}"
                )
    if optimizer_adam == 'adam':
        plt.figure(figsize=(14, 4))
        plt.plot(losses)
        plt.title("Loss")
        plt.savefig(f'{optimizer_adam}.svg')
    elif optimizer_adam == 'adam_hd':
        plt.figure(figsize=(14, 4))
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.title("Loss")
        plt.subplot(1, 2, 2)
        plt.plot(alphas)
        plt.title("α_t (learning rate)")
        plt.savefig(f'{optimizer_adam}.svg')
    else:
        plt.figure(figsize=(14, 4))
        plt.subplot(1, 3, 1)
        plt.plot(losses)
        plt.title("Loss")
        plt.subplot(1, 3, 2)
        plt.plot(alphas)
        plt.title("α_t (learning rate)")
        plt.subplot(1, 3, 3)
        plt.plot(betas1, label="β1_t")
        plt.plot(betas2, label="β2_t")
        plt.legend()
        plt.title("β_t dynamics")
        plt.savefig(f'{optimizer_adam}.svg')


if __name__ == "__main__":
    # adam, adam_hd, new_adam
    optimizer_adam = 'new_adam'

    fix_seed()
    model = Net()

    if optimizer_adam == 'adam':
        from torch.optim import Adam
        optimizer = Adam(model.parameters())
    elif optimizer_adam == 'adam_hd':
        from lib.adam_hd import AdamHD as AdamDynamic
        optimizer = AdamDynamic(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            hypergrad_lr=1e-5
        )
    else:
        from lib.new_adam import AdamDynamic
        # from lib.learning_adam import AdamDynamic
        optimizer = AdamDynamic(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            hyper_lr=(0.0, 1e-0, 0.0),
        )
    import time
    start = time.time()
    train(optimizer=optimizer, model=model, time_to_learn=10000, optimizer_adam=optimizer_adam)
    end = time.time() - start
    print(f'[INFO] Time of training: {end}')