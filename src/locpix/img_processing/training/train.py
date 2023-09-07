"""Training module.

This module contains definitions relevant to
training the model.

"""

import torch
import wandb


def train_loop(
    epochs, model, optimiser, train_loader, val_loader, loss_fn, device, model_path
):
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
            on
        model_path (string) : Where to save model to"""

    model.to(device)

    scaler = torch.cuda.amp.GradScaler()
    best_loss = 1e12
    save_epoch = 0

    for epoch in range(epochs):
        print("Epoch: ", epoch)

        # TODO : autocast look at gradient accumulation and take care with multiple gpus

        # training data
        running_train_loss = 0
        train_items = 0
        for index, data in enumerate(train_loader):
            model.train()

            img, label = data

            # note set to none is meant to have less memory footprint
            optimiser.zero_grad(set_to_none=True)

            # move data to device
            img = img.to(device)
            label = label.to(device)

            # make label same shape
            label = torch.unsqueeze(label, 1)

            # forward pass - with autocasting
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                # data.x = data.x.float()
                output = model(img)
                loss = loss_fn(output, label)
                running_train_loss += loss
                train_items += img.shape[0]

            # scales loss - calls backward on scaled loss creating scaled gradients
            scaler.scale(loss).backward()

            # unscales the gradients of optimiser then optimiser.step is called
            scaler.step(optimiser)

            # update scale for next iteration
            scaler.update()

        # val data
        # TODO: make sure torch.no_grad() somewhere
        running_val_loss = 0
        val_items = 0
        for index, data in enumerate(val_loader):
            with torch.no_grad():

                # make sure model in eval mode
                model.eval()

                img, label = data

                # note set to none is meant to have less memory footprint
                optimiser.zero_grad(set_to_none=True)

                # move data to device
                img = img.to(device)
                label = label.to(device)

                # make label same shape
                label = torch.unsqueeze(label, 1)

                # forward pass - with autocasting
                with torch.autocast(device_type="cuda"):
                    output = model(img)
                    loss = loss_fn(output, label)
                    running_val_loss += loss
                    val_items += img.shape[0]
        running_val_loss /= val_items

        wandb.log({"train loss": running_train_loss, "val loss": running_val_loss})

        if running_val_loss < best_loss:
            print("Saving best model")
            best_loss = running_val_loss
            torch.save(model.state_dict(), model_path)
            save_epoch = epoch

    # load in best model and return
    model.load_state_dict(model.state_dict())

    wandb.log({"save epoch": save_epoch})

    return model
