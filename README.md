# TED_DCMIS
The implementation of our paper "Boosting Knowledge Diversity, Accuracy, and Stability via Tri-Enhanced Distillation for
Domain Continual Medical Image Segmentation"

## Requirements
pip install -r requirements.txt

## project structure
```
   --ablation_study/
   --analysis/
   --data_prep/
   --mp/
   --storage/
   --README.md
   --requirements.txt
   --main.py
   --get.py
   --args.py
   --command
```

## Data Preparation
```
    cat data_prep/readme.md
    python data_prep/prostate_prepare.py
    python data_prep/cardiacmm_prepare.py
```

## Run
```
    cat command
    python main.py --dataset prostate --approach ted --epochs 50 \
    --experiment-name prostate-tedgugf-unet --backbone unet --device-ids 4 --gugf
```

## Analysis
```
    python analysis/eval_dataset.py # evaluate the performance of each dataset and each approach
    python analysis/table_figure.py # generate the table and figure in the paper
    python analysis/save_images.py # save the segmentation results
```

## Ablation study
```
    # ablation study of the ska
    python ablation_study/ska.py
    python ablataion_study/ska_plot.py
    # ablation study of the akt
    python ablation_study/akt.py
    python ablation_study/akt_plot.py
    # ablation study of the gugf (gvu)
    python ablation_study/gvu_prostate_plot.py
    python ablation_study/gvu_mm_plot.py
    
```

## Citation
```