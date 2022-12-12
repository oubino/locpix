"""Training module.

This module contains definitions relevant to
training the model.

"""

import torch


def train_loop(epochs, model, optimiser, train_loader, val_loader, loss_fn, device):
    """Function defining the training loop

    Args:
        epochs (int): Number of epochs to train for
        optimiser (torch.optim optimiser): Optimiser
            that controls training of the model
        model (torch geometric model): Model that
            is going to be trained
        train_loader (torch dataloader): Dataloader for the
            training dataset
        val_loader (torch dataloader): Dataloader for the
            validation dataset
        loss_fn (loss function): Loss function calculate loss
            between train and output
        device (gpu or cpu): Device to train the model
            on"""

    model.to(device)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        print("Epoch: ", epoch)

        # TODO : autocast look at gradient accumulation and take care with multiple gpus

        # training data
        for index, data in enumerate(train_loader):
            model.train()

            # note set to none is meant to have less memory footprint
            optimiser.zero_grad(set_to_none=True)

            # move data to device
            data.to(device)

            # forward pass - with autocasting
            with torch.autocast(device_type="cuda", dtype=torch.float64):
                # data.x = data.x.float()
                output = model(data)
                loss = loss_fn(output, data.y)

            # scales loss - calls backward on scaled loss creating scaled gradients
            scaler.scale(loss).backward()

            # unscales the gradients of optimiser then optimiser.step is called
            scaler.step(optimiser)

            # update scale for next iteration
            scaler.update()

        # val data
        # TODO: make sure torch.no_grad() somewhere
        for index, data in enumerate(val_loader):
            with torch.no_grad():

                # make sure model in eval mode
                model.eval()

                # note set to none is meant to have less memory footprint
                optimiser.zero_grad(set_to_none=True)

                # move data to device
                data.to(device)

                # forward pass - with autocasting
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(data)
                    loss = loss_fn(output, data.y)
