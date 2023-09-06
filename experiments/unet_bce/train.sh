#Python train script

#Run with current environment (-V) and in the current directory (-cwd)
#$ -V -cwd

#Request some time- min 15 mins - max 48 hours
#$ -l h_rt=00:15:00

# GPU/CPU request
#$ -l coproc_v100=1

#Get email at start and end of the job
#$ -m be

#Now run the job
python src/locpix/scripts/img_seg/unet_train.py -i ../../../output/c15_cells_filtered -c src/locpix/templates/unet_train_bce.yaml
