#!/bin/bash

wget -c -O data.zip https://liveln-my.sharepoint.com/:u:/g/personal/allentan_ln_hk/EQjfKJaD4dRIhUy4ZzllMT0BRLVHp_MwM237VBTkUfmptA?download=1
unzip data.zip
rm data.zip

wget -c -O checkpoint_finetune.pth https://liveln-my.sharepoint.com/:u:/g/personal/allentan_ln_hk/EULBYEXybNFDpc3gOZD1NQABCzWEjR_nlD9Rg-7USuAcfQ?download=1
mv checkpoint_finetune.pth ckpts/finetune_ckpts/


cd train/finetune
python summarisation_unimodal.py --mode test


