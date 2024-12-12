# Exploring performance decay of MMR with decreasing count of normal-shots
This section relates to the analysis of the performance decay of the MMR method introduced in the paper [Industrial Anomaly Detection with Domain Shift: A Real-world Dataset and Masked Multi-scale Reconstruction.](https://arxiv.org/abs/2304.02216).

## Performing the analysis
### Setup
#### MMR source code
Download and unzip the [`MMR.zip`](https://drive.google.com/file/d/1N4NguN8iYVykRyef6RPRhgdiAyyWVgVm/view?usp=share_link) into the `MMR` folder. This source code was taken from the official implementation accessed on 2024/12/11 from [the official GitHub repository](https://github.com/zhangzilongc/MMR). In case the official repository was updated since, we do not guarantee compatibility of our work with the updated version.
#### Pre-trained model for MMR
Download the pre-trained model of MAE (ViT-base) available at [here](https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth) or via [MMR's official GitHub repository](https://github.com/zhangzilongc/MMR).
Store the model in the MMR folder.
#### Dataset

Download the AeBAD dataset [here](https://drive.google.com/file/d/14wkZAFFeudlg0NMFLsiGwS0E593b-lNo/view?usp=share_link) or if no longer unavailable, seek newer source from [MMR's official GitHub repository](https://github.com/zhangzilongc/MMR).
After dataset extraction, they should be in the following  structure:
```
Data
|-- AeBAD
    |-- AeBAD_S
        |-- train
            |-- good
                |-- background
            |-- ...
        |-- test
            |-- ablation
                |-- background
            |-- ...
        |-- ground_truth
            |-- ablation
                |-- view
            |-- ...
    |-- AeBAD_V
        |-- ...
```
#### Virtual Environment
We strongly recommend using a virtual environment, as the method requires quite particular library versions.

If using conda, create the virtual environment `MMR-env` running the following commands in the `MMR` directory:
```
conda env create -f env.yaml
conda activate MMR-env
pip install -f https://download.pytorch.org/whl/cu113/torch_stable.html -r requirementsCustom.txt
```
Note, that we are not using the requirements.txt from the source code of MMR, as we found those to be insufficient.

After creation of the virtual environment, one more step is needed for a successful run. 
The `timm` library contains a file incompatible with one other library of the specified version.
Newer `timm` library versions are, however, incompatible with different libraries. 
Thus, we need to repair one small bug to make it compatible.
The module `timm.models.layers.helpers located` in `.../timm/models/layers` needs to be replaced with a modified [helpers.py](helpers.py) file.
For Linux, the file will, probably, be located at `~/miniconda3/envs/MMR-env/lib/python3.9/site-packages/timm/models/layers`.

#### Data Preparation and MMR Run Jobs
Move to the `MMR` folder and run 
```
bash dataPreparation.sh
```
This generates symlink-ed datasets in `Data/AeBAD_fewer_shot` for randomly generated 5 seeds for fewer normal-shots 
(training images) at multiple tiers of percentage of kept images. Method config files will also be created for each smaller dataset.

Besides this data preprocessing, `listOfRuns.sh`, `run1stHalf.sh`, `run2ndHalf.sh` files will be created. 
`listOfRuns.sh` contains the list of all runs of MMR, while `run1stHalf.sh` and `run2ndHalf.sh` contain these runs split into two to allow for running them as two `sbatch` jobs that can be run in parallel:
```
sbatch run1stHalf.sh
```
and
```
sbatch run2ndHalf.sh
```
these jobs will produce an output file, `AeBAD_MMR_1stHalf.out` and `AeBAD_MMR_2ndHalf.out` respectively.

#### Output Processing and Analysis

Filter the raw output files using 
```
python3 parseOutput.py -r AeBAD_MMR_1stHalf.out -w 1stHalf.csv
```
and
```
python3 parseOutput.py -r AeBAD_MMR_2ndHalf.out -w 2ndHalf.csv
```
These csv files can be viewed separately, or they can be combined into one using 
```
cat 1stHalf.csv > fullOutput.csv
tail -n +2 2ndHalf.csv >> fullOutput.csv
```

We include such [results file](MMR/fullOutput.csv) of our run.
