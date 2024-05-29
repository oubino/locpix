# C15 data

## Starting files

-- examples
    -- c15_data
        -- README.md
        -- config/
        -- raw/
        -- output/
            -- annotate/annotated/
            -- markers/
            -- ilastik/
                -- ilastik_pixel/
                -- ilastik_boundary/
                -- models/

## Workflow

The following assumes a familiarity with using anaconda, terminal/command-line, navigating between directories and running python scripts

Note all scripts also save the config file used inside the output folder (see below)

1. Install environment

    conda create -n locpix-env python==3.10
    conda activate locpix-env
    pip install locpix

2. Navigate to the correct directory

    cd tests/c15_data/

3. Preprocess files 

    preprocess -i raw/C15_EGFR568_EREG647 -c config/egfr_568_ereg_647_preprocess.yaml -o output -p
    preprocess -i raw/C15_EGFR647_EREG568 -c config/egfr_647_ereg_568_preprocess.yaml -o output -p
    preprocess -i raw/C15_EREG568_EGFR647 -c config/ereg_568_egfr_647_preprocess.yaml -o output -p
    preprocess -i raw/C15_EREG647_EGFR568 -c config/ereg_647_egfr_568_preprocess.yaml -o output -p

This will take files from raw/... and create folder output/preprocess/no_gt_label with preprocessed files in

It will also make a visualisation notebook in output/

4. Annotate files 

The files are already annotated inside output/annotate/annotated 

To re-annnotate run...

    annotate -i output -c config/annotate.yaml -r

You need to 

    1. Click "New labels layer"
    - Activate the paint brush
    - Draw onto membranes

    2. Click "New points layer"
    - Add points
    - Place points at each cell

This re-annotates the files if necessary (NOTE DON'T REANNOTATE IF WANT TO KEEP SAME RESULTS)

5. Segment using classic method

    classic -i output -c config/classic.yaml

Generates folder output/classic containing cell/membrane annotations

6. Install cellpose requirements (with GPU)

    pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
    git clone https://github.com/oubino/cellpose
    cd cellpose
    pip install .

7. Evaluate using untrained cellpose

    cellpose_eval -i output -c config/cellpose_eval.yaml

8. Train & segment using Cellpose 

Prepare for training 

    train_prep -i output -c config/train_prep.yaml

This creates train_files and test_files folder

    cellpose_train -i output -ct config/cellpose_train.yaml -ce config/cellpose_train_eval.yaml

This trains and evaluates the model putting output inside cellpose_train for 5 folds

9. Train & segment using UNET

    unet -i output -c config/unet_train.yaml

Generates unet/ folder with segmentations for each fold

10. Ilastik

Prepare data for Ilastik

    ilastik_prep -i output -c config/ilastik_prep.yaml

This prepares data and puts in folder ilastik/prep
We provide the ilastik project workbooks for each fold in output/ilastik/models
and their corresponding outputs in ilastik/ilastik_boundary & ilastik/ilastik_pixel

We then convert this data into the apppropriate output

    ilastik_output -i output

11. Evaluate performance on the membranes

    membrane_performance -i output -c config/membrane_performance.yaml
    membrane_performance_method -i output -c config/membrane_performance.yaml -o unet

This puts the results in folder membrane_performance

12. Visualise the results

    jupyter notebook

Open up visualisation.ipynb and run through the commands there - note most imporantly change the save_folder and save_location