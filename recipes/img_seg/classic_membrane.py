"""Classic image segmentation module

Take in items and segment using classic methods
"""

import yaml
import os
from heptapods.preprocessing import datastruc
from heptapods.visualise import vis_img
from heptapods.img_processing import watershed
import numpy as np
import pickle as pkl


if __name__ == "__main__":

    # load yaml
    with open('recipes/img_seg/classic.yaml', "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # list items
    try:
        files = os.listdir(config['input_folder'])
    except FileNotFoundError:
        raise ValueError("There should be some files to open")

    # if output directory not present create it
    if not os.path.exists(config['output_folder']):
        print('Making folder')
        os.makedirs(config['output_folder'])

    # if markers directory not present create it
    if not os.path.exists(config['markers_loc']):
        print('Making folder')
        os.makedirs(config['markers_loc'])

    for file in files:
        item = datastruc.item(None, None, None, None)
        item.load_from_parquet(os.path.join(config['input_folder'], file))

        # load in histograms
        histo_loc = os.path.join(config['input_histo_folder'], item.name + '.pkl')
        with open(histo_loc, 'rb') as f:
            histo = pkl.load(f)
        
        # segment file using watershed
        img = histo[0].T # consider only the zero channel
        log_img = vis_img.manual_threshold(img,
                                           config['vis_threshold'],
                                           how=config['vis_interpolate'])
        grey_log_img = vis_img.img_2_grey(log_img) # convert img to grey
        grey_img = vis_img.img_2_grey(img) # convert img to grey

        # img mask 
        img_mask = (grey_log_img - np.min(grey_log_img))/ \
                        (np.max(grey_log_img) - np.min(grey_log_img))

        # save mask
        save_loc = os.path.join(config['output_folder'], item.name + '.npy')
        np.save(save_loc,img_mask)


"""
        # threshold it 
        if cfg['classic']['prob_threshold'] == 'recall':
            pred_df 

            prob_threshold = ...
        semantic_mask = np.where(semantic_mask>prob_threshold,1,0)

        
        # tested very small amount annd line below is better than doing watershed on grey_log_img
        instance_mask = classical.watershed_segment(grey_img, coords=markers, plot=False) # watershed on the grey image
        
        # save
        np.save(output_markers, markers) # save marker coordinates
        np.save(output_histo, instance_mask) # save instance mask
        # save to csv the SEMANTIC MASK (0 background, 1 membranes)
        # note histo_mask so have to transpose the semantic mask which is in image space
        data_item.save_labels_to_csv(output_coords, drop_zero_label=False, histo_mask=semantic_mask.T)

        # save images
        # save boundary image
        vis.visualise_boundaries(data_item.img, instance_mask, data_item.bin_sizes, channels=[0], threshold=vis_threshold, how=vis_interpolation, origin=origin_loc, save=True, save_loc=output_boundary_img)

        # load and save seg histos in 4 colours - consider only zero channel
        vis.visualise_seg(data_item.img, in`stance_mask, data_item.bin_sizes, channels=[0], threshold=vis_threshold, how=vis_interpolation, origin=origin_loc, save=True, save_loc=output_histo_img, four_colour=True)
"""
