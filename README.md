## PixaR: Segmenting Large Multimodal Model for Region-level Retrieval-Augmented Generation
This repository contains the code for the paper [PixaR: Segmenting Large Multimodal Model for Region-level Retrieval-Augmented Generation](https://arxiv.org/abs/2305.XXXXX).

## Requirements
Install the required packages and dependencies with the following command:
```bash
conda create --name segrag --file requirements.txt
```

## Data Setup
The following datasets are used to construct the training dataset.

## Training
```bash
chmod +x scripts/train_segrag.sh
./train_segrag.sh
```

## Evaluation
```bash
chmod +x scripts/eval_segrag.sh
./eval_segrag.sh
```
