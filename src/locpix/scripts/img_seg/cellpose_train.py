#!/usr/bin/env python
"""Cellpose segmentation module

Take in items and train the Cellpose module
"""

import yaml
import tkinter as tk
from tkinter import filedialog
import torch
from torch.utils.data import DataLoader
from locpix.img_processing.data_loading import dataset
from locpix.img_processing.training import train
import os
from torchvision import transforms
from cellpose import models
from torchsummary import summary
import argparse

# from locpix.scripts.img_seg import cellpose_train_config


def main():

    # Load in config

    parser = argparse.ArgumentParser(description="Cellpose")
    parser.add_argument(
        "-i",
        "--project_directory",
        action="store",
        type=str,
        help="the location of the project directory",
    )
    parser.add_argument(
        "-c",
        "--config",
        action="store",
        type=str,
        help="the location of the .yaml configuaration file\
                             for preprocessing",
    )

    args = parser.parse_args()

    # input project directory
    if args.project_directory is not None:
        project_folder = args.project_directory
    else:
        root = tk.Tk()
        root.withdraw()
        project_folder = filedialog.askdirectory(title="Project directory")

    if args.config is not None:
        # load yaml
        with open(args.config, "r") as ymlfile:
            config = yaml.safe_load(ymlfile)
            # cellpose_train_config.parse_config(config)
    else:
        root = tk.Tk()
        root.withdraw()
        # gt_file_path = filedialog.askdirectory()
        # config = cellpose_train_config.config_gui(gt_file_path)

    # load in config
    input_root = os.path.join(project_folder, "annotate/annotated")
    #label_root = os.path.join(project_folder, "annotate/annotated")
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    gpu = config["gpu"]
    optimiser = config["optimiser"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    num_workers = config["num_workers"]
    loss_fn = config["loss_fn"]
    train_files = config["train_files"]

    # list items
    try:
        files = os.listdir(input_root)
        files = [os.path.splitext(file)[0] for file in files]
    except FileNotFoundError:
        raise ValueError("There should be some files to open")

    # make necessary folders if not present
    preprocessed_folder = os.path.join(project_folder, "cellpose_train")
    if os.path.exists(preprocessed_folder):
        raise ValueError(f"Cannot proceed as {preprocessed_folder} already exists")
    else:
        os.makedirs(preprocessed_folder)

    print("files", files)

    # define gpu or cpu
    # if data is on gpu then don't need to pin memory
    # and this causes errors if try
    if gpu is True and not torch.cuda.is_available():
        raise ValueError(
            "No gpu available, can run on cpu\
                         instead"
        )
    elif gpu is True and torch.cuda.is_available():
        device = torch.device("cuda")
        pin_memory = False
    elif gpu is False:
        device = torch.device("cpu")
        pin_memory = True
    else:
        raise ValueError("Specify cpu or gpu !")

    # split files into train and validation
    train_files = files[0:5]
    val_files = files[5:-1]

    # check train and test files
    print("Train files")
    print(train_files)

    # define transformations for train, test
    train_transform = [transforms.ToTensor()]
    val_transform = [transforms.ToTensor()]

    # Initialise train and val dataset
    train_set = dataset.ImgDataset(
        input_root, train_files, ".parquet", train_transform
    )
    val_set = dataset.ImgDataset(
        input_root, val_files, ".parquet", val_transform
    )

    print('Preprocessing datasets')

    # Pre-process train and val dataset
    train_set.preprocess(os.path.join(preprocessed_folder, "train"))
    val_set.preprocess(os.path.join(preprocessed_folder, "val"))

    # initialise dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    # intialise model
    model = models.CellposeModel(model_type=config["model"])

    # cellpose need to freeze layers/ train on intermediate output etc.

    # initialise optimiser
    if optimiser == "adam":
        optimiser = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    # initialise loss function
    if loss_fn == "nll":
        loss_fn = torch.nn.functional.nll_loss

    # print parameters
    print("\n")
    print("---- Params -----")
    print("\n")
    print("Input features: ", train_set.num_node_features)
    print("Num classes: ", train_set.num_classes)
    print("Batch size: ", batch_size)
    print("Epochs: ", epochs)

    # model summary
    print("\n")
    print("---- Model summary ----")
    print("\n")
    number_nodes = 1000  # this is just for summary, has no bearing on training
    summary(
        model,
        input_size=(train_set.num_node_features, number_nodes),
        batch_size=batch_size,
    )

    # train loop
    print("\n")
    print("---- Training... ----")
    print("\n")
    train.train_loop(
        epochs, model, optimiser, train_loader, val_loader, loss_fn, device
    )
    print(
        "Need checks here to make sure model weights are\
          correct"
    )
    print("\n")
    print("---- Finished training... ----")
    print("\n")

    # save final model
    print("\n")
    print("---- Saving final model... ----")
    print("\n")


"""
    # ---------------------#

    for file in files:
        item = datastruc.item(None, None, None, None)
        item.load_from_parquet(os.path.join(config["input_folder"], file))

        # load in histograms
        histo_loc = os.path.join(config["input_histo_folder"], item.name + ".pkl")
        with open(histo_loc, "rb") as f:
            histo = pkl.load(f)

        # ---- segment membranes ----

        if config["sum_chan"] is False:
            img = histo[0].T  # consider only the zero channel
        elif config["sum_chan"] is True:
            img = histo[0].T + histo[1].T
        else:
            raise ValueError("sum_chan should be true or false")
        img = vis_img.manual_threshold(
            img, config["vis_threshold"], how=config["vis_interpolate"]
        )
        imgs = [img]
        model = models.CellposeModel(model_type=config["model"])
        channels = config["channels"]
        # note diameter is set here may want to make user choice
        # doing one at a time (rather than in batch) like this might be very slow
        _, flows, _ = model.eval(imgs, diameter=config["diameter"], channels=channels)
        # flows[0] as we have only one image so get first flow
        # flows[0][2] as this is the probability see
        # (https://cellpose.readthedocs.io/en/latest/api.html)
        semantic_mask = flows[0][2]

        # convert mask (probabilities) to range 0-1
        semantic_mask = (semantic_mask - np.min(semantic_mask)) / (
            np.max(semantic_mask) - np.min(semantic_mask)
        )

        # ---- segment cells ----
        # get markers
        markers_loc = os.path.join(config["markers_loc"], item.name + ".npy")
        try:
            markers = np.load(markers_loc)
        except FileNotFoundError:
            raise ValueError(
                "Couldn't open the file/No markers were found in relevant location"
            )

        # tested very small amount annd line below is better than
        # doing watershed on grey_log_img
        instance_mask = watershed.watershed_segment(
            semantic_mask, coords=markers
        )  # watershed on the grey image

        # ---- save ----

        # save membrane mask
        save_loc = os.path.join(config["output_membrane_prob"], item.name + ".npy")
        np.save(save_loc, semantic_mask)

        # save markers
        np.save(markers_loc, markers)

        # save instance mask to dataframe
        df = item.mask_pixel_2_coord(instance_mask)
        item.df = df
        item.save_to_parquet(
            config["output_cell_df"], drop_zero_label=False, drop_pixel_col=True
        )

        # save cell segmentation image - consider only zero channel
        imgs = {key: value.T for (key, value) in histo.items()}
        save_loc = os.path.join(config["output_cell_img"], item.name + ".png")
        vis_img.visualise_seg(
            imgs,
            instance_mask,
            item.bin_sizes,
            channels=[0],
            threshold=config["vis_threshold"],
            how=config["vis_interpolate"],
            blend_overlays=True,
            alpha_seg=0.5,
            origin="upper",
            save=True,
            save_loc=save_loc,
            four_colour=True,
        )

        # save yaml file
        yaml_save_loc = os.path.join(project_folder, "cellpose_train.yaml")
        with open(yaml_save_loc, "w") as outfile:
            yaml.dump(config, outfile)
"""

if __name__ == "__main__":
    main()
