import torch


def train(model, data, lr):
    losses = []
    for epoch in range(10):
        for batch in data:
            output = model(batch["x"])
            loss = torch.nn.functional.mse_loss(output, batch["y"])
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()
            losses.append(loss.item())
            print("loss:", loss)
    return losses


def main():
    print("Starting training...")
    from torch import nn

    model = nn.Linear(10, 1)
    model.optimizer = torch.optim.SGD(model.parameters(), 0.01)
    # data = [{"x": torch.randn(10), "y": torch.randn(1)} for i in range(100)]
    print("done")


if __name__ == "__main__":
    main()
