import argparse
import sys

import torch
import click
import torchmetrics

from data import mnist
from model import MyAwesomeModel


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
@click.option("--epochs", default=1, help='number of epochs to train')
def train(lr, epochs):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=16)
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    loss_fn = torch.nn.NLLLoss()

    for epoch in range(epochs):
        print("EPOCH: {:d}".format(epoch))
        for i, Xy in enumerate(train_dataloader):
            X, y = Xy
            logits = model(X)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%100 == 0:
                print(loss.item())

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, "checkpoints/checkpoint_{:d}".format(epoch))


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    checkpoint = torch.load(model_checkpoint)
    model = MyAwesomeModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    _, test_set = mnist()

    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=16)
    predictions = []
    true = []
    for X, y in test_dataloader:
        predictions.append(model(X))
        true.append(y)
    predictions = torch.vstack(predictions)
    true = torch.hstack(true)
    top1_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
    print(top1_acc(predictions, true))

cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    