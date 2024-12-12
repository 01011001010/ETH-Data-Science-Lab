#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "No arguments supplied, please specity the global path to the directory containing Data and MMR folders"
    exit 1
fi

if [ $# -gt 1 ]
  then
    echo "Too many arguments"
    exit 1
fi

mkdir -p "${1}/Data/AeBAD_fewer_shot"

[ -f toDel.txt ] && rm toDel.txt
[ -f listOfRuns.sh ] && rm listOfRuns.sh

for percentage in 1 3 5 10 15 25 35 45 55 70 85 100
do
    for seed in $RANDOM $RANDOM $RANDOM $RANDOM $RANDOM
    do
        echo processing $percentage % with seed $seed
        folderName="${percentage}_percent_${seed}_seed"
        python3 "${1}/Data/generateBashForSymplinkCopies.py" -n "$folderName" -p $1 > generate.sh
        bash generate.sh
        python3 "${1}/Data/generateListForDeletion.py" -k $percentage -n "$folderName" -p $1 >> toDel.txt
        echo "RNG_SEED: ${seed}

OUTPUT_DIR: './log_MMR_AeBAD_S_${folderName}'


DATASET:
  name: 'aebad_S'
  resize: 256
  imagesize: 224
  subdatasets: ['AeBAD_S']
  domain_shift_category: 'same'

TRAIN:
  enable: True
  save_model: False
  method: 'MMR'
  dataset_path: '${1}/Data/AeBAD_fewer_shot/${folderName}'
  backbone: 'wideresnet50'

  MMR:
    DA_low_limit: 0.7
    DA_up_limit: 1.
    layers_to_extract_from : ['layer1', 'layer2', 'layer3']
    feature_compression : False
    scale_factors : (4.0, 2.0, 1.0)
    FPN_output_dim : (256, 512, 1024)
    load_pretrain_model : True
    model_chkpt : '${1}/MMR/mae_visualize_vit_base.pth'
    finetune_mask_ratio : 0.4
    test_mask_ratio : 0.


TRAIN_SETUPS:
  batch_size: 16
  num_workers: 2
  learning_rate: 0.001
  epochs: 200
  weight_decay: 0.05
  warmup_epochs: 50

TEST:
  enable: True
  save_segmentation_images: False
  method: 'MMR'
  dataset_path: '${1}/Data/AeBAD_fewer_shot/${folderName}'

  VISUALIZE:
    Random_sample: True

TEST_SETUPS:
  batch_size: 32" > "${1}/Data/AeBAD_fewer_shot/${folderName}/config.yaml"
    
    echo python3 main.py --device "0" --cfg "${1}/Data/AeBAD_fewer_shot/${folderName}/config.yaml" --opts NUM_GPUS 1 RNG_SEED ${seed} OUTPUT_DIR "'./log_MMR_AeBAD_S_${folderName}'" >> listOfRuns.sh


    done        
done

xargs rm < toDel.txt

#rm toDel.txt
#rm generate.sh


sed -ne 'w odd.txt' -e 'n; w even.txt' listOfRuns.sh


[ -f run1stHalf.sh ] && rm run1stHalf.sh
[ -f run2ndHalf.sh ] && rm run2ndHalf.sh

cat jobHeader.txt > run1stHalf.sh
cat jobHeader.txt > run2ndHalf.sh
echo AeBAD_MMR_1stHalf.out >> run1stHalf.sh
echo AeBAD_MMR_2ndHalf.out >> run2ndHalf.sh
echo -e '\n' >> run1stHalf.sh
echo -e '\n' >> run2ndHalf.sh
cat odd.txt >> run1stHalf.sh
cat even.txt >> run2ndHalf.sh

rm odd.txt
rm even.txt

