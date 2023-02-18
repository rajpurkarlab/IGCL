# Learning Generalized Medical Image Representations through Image-Graph Contrastive Pretraining

> Over the past few years, artificial intelligence has led to the
> automation of increasingly complex medical image interpretation tasks,
> such as disease diagnosis, sometimes even exceeding the performance of
> medical experts. Foundation models like BERT and CLIP pretrained on
> large datasets have advanced state-of-the-art performance across a
> diverse set of downstream image and language benchmarks. In
> particular, existing work has demonstrated the effectiveness of
> contrastive learning in producing high quality medical image
> representations, allowing for highly label-efficient task-specific
> finetuning. Using chest x-rays as a driving example, in this paper, we
> show that through image-graph contrastive learning with paired chest
> x-rays and report knowledge graphs, we can obtain even higher quality
> image representations. Specifically, we show that at no additional
> data annotation cost over existing methods, our method outperforms
> existing state-of-the-art pretraining methods on 4 out of 5 benchmark
> tasks for CheXpert, a multi-label pathology classification dataset of
> chest x-rays. This work demonstrates the ability of structured
> knowledge graph embeddings to guide learning of salient image
> features.


# Conda Environments

Our models are trained using a DRACON graph encoder built on a PyTorch Geometric backbone.
Evaluation is executed via modified version of the CXR-Learn platform.

Requirement files for training and evaluation can be found in `reqs/imggraph_env.txt` and `reqs/cxrlearn_env.txt` files respectively.

## Training

An example training script is provided in `src/train.sh`. To begin training, first modify model parameters as you see fit. Note that for proper execution of the code on your system of choice, you will need to modify the path params relating to `graph_root`,  `raw_image_path`,  `processed_image_path`, and `processed_graph_path`.

Scripts relating to constructing the required files from the MIMIC-CXR and RadGraph datasets can be found in `src/construct_raw_data` and `src/dataset`.

## Finetuning + Evaluation

An example evlauation script is provided in `src/exec.sh`. This follows the procedure used by the CXR-Learn platform. To begin training, first modify model parameters as you see fit. Note that for proper execution of the code on your system of choice, you will need to modify the path params relating to `downstream_path`.

At the end of execution, you will find a pickled python dictionary saved that contains `config`, `auroc`, `mcc`, `auc_intervals`, and `mcc_intervals` results from executing the evaluation experiment.
