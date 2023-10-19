import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

class ConsistencyWeight(nn.Module):
    def __init__(self, max_weight, max_epoch, ramp='sigmoid'):
        super(ConsistencyWeight, self).__init__()
        self.max_weight = max_weight
        self.max_epoch = max_epoch - 25
        self.ramp = ramp

    def forward(self, epoch):
        if epoch <= 25:
            return 0.0
        current = np.clip(epoch, 0.0, self.max_epoch)
        phase = 1.0 - current / self.max_epoch
        if self.ramp == 'sigmoid':
            ramps = float(np.exp(-5.0 * phase * phase))
        elif self.ramp == 'log':
            ramps = float(1 - np.exp(-5.0 * current / self.max_epoch))
        elif self.ramp == 'exp':
            ramps = float(np.exp(5.0 * (current / self.max_epoch - 1)))
        else:
            ramps = 1.0

        consistency_weight = self.max_weight * ramps
        return consistency_weight

def main():
    max_weight = 2.0
    max_epoch = 130

    consistency_weights = []
    for epoch in range(25, max_epoch + 1):
        consistency_weight = ConsistencyWeight(max_weight, max_epoch)
        weight = consistency_weight(epoch)
        consistency_weights.append(weight)
        print(epoch, weight)

    epochs = list(range(25, max_epoch + 1))

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, consistency_weights, marker='o', linestyle='-', color='b')
    plt.title("Consistency Weights for Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Consistency Weight")
    plt.grid(True)
    plt.ylim(0, max_weight + 0.1)
    plt.show()


if __name__ == "__main__":
    main()