#!/usr/bin/env python
"""UNET segmentation module

Take in items and trains UNET
"""

import yaml
import os
import argparse
import json
import time
from locpix.img_processing.models.unet import two_d_UNet
from locpix.scripts.img_seg import img_train_prep
from locpix.img_processing.training import train, loss
from locpix.img_processing.data_loading import dataset
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from locpix.img_processing import watershed
from locpix.preprocessing import datastruc
import wandb


def main():

    # Load in config

    parser = argparse.ArgumentParser(description="Train UNET.")
    parser.add_argument(
        "-i",
        "--project_directory",
        action="store",
        type=str,
        help="the location of the project directory",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--config",
        action="store",
        type=str,
        help="the location of the .yaml configuaration file\
                             for preprocessing",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--project_metadata",
        action="store_true",
        help="check the metadata for the specified project and" "seek confirmation!",
    )

    args = parser.parse_args()

    project_folder = args.project_directory
    # load config
    with open(args.config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    metadata_path = os.path.join(project_folder, "metadata.json")
    with open(
        metadata_path,
    ) as file:
        metadata = json.load(file)
        # check metadata
        if args.project_metadata:
            print("".join([f"{key} : {value} \n" for key, value in metadata.items()]))
            check = input("Are you happy with this? (YES)")
            if check != "YES":
                exit()
        # add time ran this script to metadata
        file = os.path.basename(__file__)
        if file not in metadata:
            metadata[file] = time.asctime(time.gmtime(time.time()))
        else:
            print("Overwriting metadata...")
            metadata[file] = time.asctime(time.gmtime(time.time()))
        with open(metadata_path, "w") as outfile:
            json.dump(metadata, outfile)

    # for every fold
    folds = len(metadata["train_folds"])

    # make folder
    time_o = time.gmtime(time.time())
    time_o = (
        f"time_{time_o[3]}_{time_o[4]}_{time_o[4]}"
        f"_date_{time_o[2]}_{time_o[1]}_{time_o[0]}"
    )
    # model_save_path = os.path.join(project_folder, "unet_train", time_o)
    # folders = [
    #    model_save_path,
    # ]
    # for folder in folders:
    #    if os.path.exists(folder):
    #        raise ValueError(f"Cannot proceed as {folder} already exists")
    #    else:
    #        os.makedirs(folder)

    print("------ Training --------")

    for fold in range(folds):

        print(f"----- Fold {fold} -------")

        # make folder
        model_save_path = os.path.join(
            project_folder, f"{config['folder_name']}/models/{fold}"
        )
        if os.path.exists(model_save_path):
            raise ValueError(f"Cannot proceed as {model_save_path} already exists")
        else:
            os.makedirs(model_save_path)

        # image train prep
        img_train_prep.preprocess_train_files(
            project_folder, config, metadata, fold, f"{config['folder_name']}"
        )

        # train model
        train_folder = os.path.abspath(
            os.path.join(project_folder, f"train_files/{config['folder_name']}/train")
        )
        val_folder = os.path.abspath(
            os.path.join(project_folder, f"train_files/{config['folder_name']}/val")
        )

        lr = config["learning_rate"]
        wd = config["weight_decay"]
        epochs = config["epochs"]
        batch_size = config["batch_size"]
        num_workers = config["num_workers"]
        tf = config["train_transforms"]
        if config["use_gpu"] is True and torch.cuda.is_available():
            device = torch.device("cuda")
            pin_memory = False
        elif config["use_gpu"] is False:
            pin_memory = True
            device = torch.device("cpu")
        else:
            raise ValueError("Specify cpu or gpu !")
        print("Device: ", device)

        # initialise model and optimiser
        model = two_d_UNet(1, 1, bilinear=False)
        optimiser = Adam(model.parameters(), lr=lr, weight_decay=wd)

        # init transforms, dataset and dataloader
        train_files = metadata["train_folds"][fold]
        val_files = metadata["val_folds"][fold]
        train_dataset = dataset.ImgDataset(
            train_folder,
            train_files,
            tf,
            train=True,
        )
        val_dataset = dataset.ImgDataset(
            val_folder,
            val_files,
            tf,
            train=False,
            mean=train_dataset.mean,
            std=train_dataset.std,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )

        # initialise loss function
        if config["loss_fn"] == "bce":
            print("Using BCE loss!")
            loss_fn = torch.nn.BCEWithLogitsLoss()
        elif config["loss_fn"] == "dice":
            print("Using DICE loss")
            loss_fn = loss.dice_loss()
        else:
            raise ValueError("Loss function must be specified in config")

        # initialise wandb
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=config["wandb_project"],
            # track hyperparameters and run metadata
            config={
                "learning_rate": lr,
                "dataset": config["wandb_dataset"],
                "epochs": epochs,
                "fold": fold,
            },
        )

        # train loop
        model = train.train_loop(
            epochs,
            model,
            optimiser,
            train_loader,
            val_loader,
            loss_fn,
            device,
            os.path.join(model_save_path, f"{time_o}_model.pt"),
        )

        # need to save mean and std from normalisation in model path
        mean = torch.tensor(train_dataset.mean)
        std = torch.tensor(train_dataset.std)
        np.save(os.path.join(model_save_path, f"{time_o}_mean.npy"), mean)
        np.save(os.path.join(model_save_path, f"{time_o}_std.npy"), std)

        # clean up
        img_train_prep.clean_up(project_folder, f"{config['folder_name']}")

        # stop wandb
        wandb.finish()

    print("------ Outputting for evaluation -------- ")

    # prep images
    img_train_prep.preprocess_all_files(
        project_folder, config, metadata, f"{config['folder_name']}"
    )

    img_folder = os.path.join(
        project_folder, f"train_files/{config['folder_name']}/all"
    )

    # create dataset
    train_files = metadata["train_files"]
    test_files = metadata["test_files"]
    all_files = train_files + test_files

    tf = config["test_transforms"]

    for fold in range(folds):

        model_folder = os.path.join(
            project_folder, f"{config['folder_name']}/models/{fold}"
        )
        mean_path = os.path.join(model_folder, f"{time_o}_mean.npy")
        std_path = os.path.join(model_folder, f"{time_o}_std.npy")

        img_dataset = dataset.ImgDataset(
            img_folder,
            all_files,
            tf,
            train=False,
            mean=np.load(mean_path),
            std=np.load(std_path),
        )

        # output folder
        cell_seg_df_folder = os.path.join(
            project_folder, f"{config['folder_name']}/{fold}/cell/seg_dataframes"
        )
        cell_seg_img_folder = os.path.join(
            project_folder, f"{config['folder_name']}/{fold}/cell/seg_img"
        )
        cell_memb_folder = os.path.join(
            project_folder, f"{config['folder_name']}/{fold}/membrane/prob_map"
        )
        folders = [
            cell_memb_folder,
            cell_seg_df_folder,
            cell_seg_img_folder,
        ]
        for folder in folders:
            if os.path.exists(folder):
                raise ValueError(f"Cannot proceed as {folder} already exists")
            else:
                os.makedirs(folder)

        # run through model

        for index in range(img_dataset.__len__()):
            with torch.no_grad():

                # get img
                img, label = img_dataset.__getitem__(index)

                # expand img dimensions so has batch of 1?
                img_name = img_dataset.input_data[index]
                img_name = os.path.basename(img_name)
                img_name = os.path.splitext(img_name)[0]
                img = torch.unsqueeze(img, 0)

                # make sure model in eval mode
                model.eval()

                # note set to none is meant to have less memory footprint
                optimiser.zero_grad(set_to_none=True)

                # move data to device
                img = img.to(device)
                # label = label.to(device)

                # forward pass - with autocasting
                with torch.autocast(device_type="cuda"):
                    output = model(img)
                    output = torch.sigmoid(output)

            # reduce batch and channel dimensions
            output = torch.squeeze(output, 0)
            output = torch.squeeze(output, 0)

            # convert tensor to numpy
            output = output.cpu().numpy()

            # ---- segment cells ----
            # get markers
            markers_loc = os.path.join(project_folder, "markers")
            markers_loc = os.path.join(markers_loc, img_name + ".npy")
            try:
                markers = np.load(markers_loc)
            except FileNotFoundError:
                raise ValueError(
                    "Couldn't open the file/No markers were found in relevant location"
                )

            # tested very small amount annd line below is better than doing
            # watershed on grey_log_img
            instance_mask = watershed.watershed_segment(
                output, coords=markers
            )  # watershed on the grey image

            # ---- save ----

            # save membrane mask
            save_loc = os.path.join(cell_memb_folder, img_name + ".npy")
            np.save(save_loc, output)

            # save instance mask to dataframe
            item_path = os.path.join(
                project_folder, "annotate/annotated", img_name + ".parquet"
            )
            item = datastruc.item(None, None, None, None, None)
            item.load_from_parquet(item_path)
            df = item.mask_pixel_2_coord(instance_mask)
            item.df = df
            item.save_to_parquet(
                cell_seg_df_folder, drop_zero_label=False, drop_pixel_col=True
            )

            # save cell segmentation image (as .npy) - consider only one channel
            save_loc = os.path.join(cell_seg_img_folder, item.name + ".npy")
            np.save(save_loc, instance_mask)

    # clean up
    img_train_prep.clean_up_all(project_folder, f"{config['folder_name']}")

    # save train yaml file
    yaml_save_loc = os.path.join(project_folder, f"{config['folder_name']}.yaml")
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
