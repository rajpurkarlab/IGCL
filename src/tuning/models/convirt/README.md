Transferring Knowledge from Medical Text Understanding to Medical Image Understanding
==================

This repo contains code to study knowledge transfer from medical text understanding to medical image understanding, via contrastive sampling-based pretraining methods.

## Data Preparation

### Data for pretraining

Currently we use the MIMIC-CXR chest X-ray image and report data for the pretraining task. Link the directory on the cluster to your local code directory with:

```bash
ln -s /u/scr/zyh/develop/data-open/mimic-cxr-jpg-resized dataset/mimic-cxr
```

Inside the linked folder, the `mimic-cxr/files` directory contains all image data; `mimic-cxr/meta.json` contains some converted meta data in json format; `reports_tokenized.json` contains the tokenized reports for each image (reports are split into two sections, "findings" and "impression").

### Data for CheXpert classification

Link the CheXpert directory on the cluster to the local directory with:

```bash
ln -s /u/scr/zyh/develop/data-open/CheXpert-v1.0-small dataset/chexpert
```

### Data for RSNA pneumonia detection

Link the RSNA pneumonia data to the local directory with:

```bash
ln -s /u/scr/zyh/develop/data-open/rsna-pneumonia-jpg dataset/rsna
```


## Running image training and evaluation

The following example python code shows how to train a ResNet50 model on the CheXpert task, save the model to folder `saved_model/01`, and run evaluation.

```python
python ./run_chexpert.py --model_name resnet50 --id 01
python ./run_chexpert.py --mode eval --id 01
```
