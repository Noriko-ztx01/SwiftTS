# SwiftTS

## Introduction
This is the official PyTorch code for "SwiftTS: A Swift Selection Framework for Time Series Pre-trained
 Models via Multi-task Meta-Learning".

We propose SwiftTS, the first method to utilize a learning-guided approach for identifying and ranking the time series pre-trained models. 
It collects the performance of various dataset-model pairs across different horizons to learn how models perform on different datasets, enabling performance prediction on unseen target datasets. 
SwiftTS features a lightweight dual-encoder architecture that embeds time series and candidate models with rich characteristics, computing patchwise compatibility scores between data and model embeddings for efficient selection. 
To further enhance the generalization across datasets and horizons, we introduce a horizon-adaptive expert composition module that dynamically adjusts expert weights based on the horizon. 
Moreover, a transferable cross-task learning strategy with cross-dataset and cross-horizon task sampling is adopted to improve out-of-distribution (OOD) robustness.

## Quick Start
## Installation
First clone the repository and then enter the project directory. The location and description of all files in this project are provided in the final `Complete Directory Structure` section.

The project dependencies can be installed by executing the following script in the root of the repository:
```bash
conda create --name SwiftTS python=3.8
conda activate SwiftTS
pip install -r requirements.txt
```

## Prepaer Datasets

You can obtain the well pre-processed downstream datasets from [Google Drive](https://drive.google.com/file/d/1ZrDotV98JWCSfMaQ94XXd6vh0g27GIrB/view) and place them in the directory `./datasets/forecasting`.

Here, for convenience, the model's meta-information has been placed in `./data_provider/model.xlsx`, and the model's topological embeddings and functional embeddings have already been placed in `./datasets/embedding`.

Then you can use the `./data_provider/datasets_make.py` file to generate the meta_datasets.

Moreover, for ease of use, we have already placed the final processed meta_datasets in the `./datasets/full_all directory`. 
Therefore, you do not need to perform the above dataset processing steps and can directly run the training and evaluation code.

## Run
### Train 
If you want to perform training, run the following command:
```bash
python run.py \
    --inner_lr "$inner_lr" \
    --meta_lr "$meta_lr" \
    --mse_weight "$mse_weight" \
    --train_epochs "$train_epochs" \
    --save_path "$save_path" \
    --seed "$seed" \
    --setting "$setting" \
    --horizon_setting "$horizon_setting" \
    --k_shot "$k_shot" \
    --n_query "$n_query" \
    --is_training True
```

### Evaluate 
After training, if you want to perform evaluation, set `is_training` to False and run:
```bash
python run.py \
    --inner_lr "$inner_lr" \
    --meta_lr "$meta_lr" \
    --mse_weight "$mse_weight" \
    --train_epochs "$train_epochs" \
    --save_path "$save_path" \
    --seed "$seed" \
    --setting "$setting" \
    --horizon_setting "$horizon_setting" \
    --k_shot "$k_shot" \
    --n_query "$n_query" \
    --is_training False
```


### Script Configuration
We have also included the relevant parameter configurations in `run.sh`. To execute the training script with default settings, simply run:
```bash
bash run.sh
```

> Note: The `run.sh` script sets up the environment variables and runs the training command automatically. 
You can modify the parameters inside run.sh to customize the training or evaluation process.

## Detailed guide

#### Training Parameters

- **zoo_size**: Size of zoo for meta input, likely indicating the number of models or configurations considered in a "model zoo".
- **train_datasets**: List of datasets used for training.
- **val_datasets**: List of datasets used for validation.
- **test_datasets**: List of datasets used for testing.
- **train_epochs**: Number of training epochs, defining how many times the entire dataset will be passed forward and backward through the model during training.
- **mse_weight**: MSE weight, the weighting factor applied to the Mean Squared Error loss component during training.
- **rank_weight**: Rank weight, indicating the importance given to ranking loss in the overall loss function.
- **meta_batch_size**: Number of tasks sampled per epoch, relevant in meta-learning setups like MAML.
- **k_shot**: Number of support samples per task, indicating how many examples are available for adaptation within each task.
- **n_query**: Number of query samples per task, the number of samples used to evaluate performance after adapting to a task.
- **inner_lr**: Learning rate for inner loop updates, specifically for updating model parameters during adaptation phase in meta-learning.
- **meta_lr**: Learning rate for outer loop (meta-optimizer), used for updating the meta-parameters based on the results of the inner loops.
- **num_inner_updates**: Number of inner loop updates per task, specifying how many gradient descent steps are taken during the adaptation phase.
- **d_model**: Model dimension, often used in transformer architectures to denote the dimensionality of the model's hidden layers.
- **patience**: Patience for early stopping, the number of epochs with no improvement after which training will be stopped.
- **seed**: Random seed, used for initializing random number generators to ensure reproducibility.
- **num_experts**: Number of experts, potentially referring to the number of sub-models or components in an ensemble or mixture-of-experts setup.
- **topo_dim**: Topological dimension, possibly relating to the dimensions of topological features in the model.
- **func_dim**: Functional dimension, likely referring to the dimensions of functional features or embeddings.
- **meta_dim**: Meta dimension, indicating the dimensionality of meta features used in the model.
- **meta_outdim**: Meta output dimension, specifying the dimension of the output from the meta part of the model.
- **input_dim**: Input dimension for meta, representing the dimension of the input vector for meta-learning components.
- **d_meta_dim**: Dimension for meta input, possibly referring to a specific dimension related to meta learning inputs.
- **setting**: Zero-shot or full-shot, indicating whether the model is trained in a zero-shot learning setting or a full-shot one.
- **horizon_setting**: Horizon settings, choices include 'all', '96', '192', '336', '720', denoting different prediction horizons.
- **is_training**: Enable training mode, a flag indicating if the model should operate in training mode.
- **save_path**: Path where the model and logs are saved.

- **seq_len**: Historical window size, indicating how many past time steps are used as input.
- **pred_len**: Prediction length, representing the number of future time steps to predict.
- **patch_size**: Patch size, used in models that process data in patches (e.g., image or sequence patches).
- **num_heads**: Number of attention heads in the Transformer model, which allows the model to focus on different parts of the input simultaneously.
- **num_layers**: Number of layers in the Transformer model, affecting the depth and representational power of the model.
- **embed_dim**: Embedding dimension, defining the size of the feature vectors produced by the embedding layer.
- **in_channels**: Channels, typically referring to the number of input channels in the data (e.g., 1 for grayscale images).

Complete Directory Structure
```bash
SwiftTS
├─ datasets
│  ├─ embedding   # Model embeddings
│  │  ├─ Chronos_func_emb.npy
│  │  ├─ Chronos_topo_emb.npy
│  │  ├─ Moirai_func_emb.npy
│  │  ├─ Moirai_topo_emb.npy
│  │  ├─ Moment_func_emb.npy
│  │  ├─ Moment_topo_emb.npy
│  │  ├─ RoseModel_func_emb.npy
│  │  ├─ RoseModel_topo_emb.npy
│  │  ├─ TimerModel_func_emb.npy
│  │  ├─ TimerModel_topo_emb.npy
│  │  ├─ TimesFM_func_emb.npy
│  │  ├─ TimesFM_topo_emb.npy
│  │  ├─ TinyTimeMixer_func_emb.npy
│  │  ├─ TinyTimeMixer_topo_emb.npy
│  │  ├─ UniTS_func_emb.npy
│  │  └─ UniTS_topo_emb.npy
│  ├─ forecasting
│  ├─ full_all  # Meta-datasets
│  │  ├─ AQShunyi_full_all.pkl
│  │  ├─ CzeLan_full_all.pkl
│  │  ├─ Electricity_full_all.pkl
│  │  ├─ ETTh1_full_all.pkl
│  │  ├─ ETTh2_full_all.pkl
│  │  ├─ ETTm1_full_all.pkl
│  │  ├─ ETTm2_full_all.pkl
│  │  ├─ Exchange_full_all.pkl
│  │  ├─ PEMS08_full_all.pkl
│  │  ├─ Solar_full_all.pkl
│  │  ├─ Traffic_full_all.pkl
│  │  ├─ Weather_full_all.pkl
│  │  ├─ Wind_full_all.pkl
│  │  └─ ZafNoo_full_all.pkl
│  └─ __init__.py
├─ data_provider
│  ├─ dataset.xlsx
│  ├─ datasets_make.py
│  ├─ meta_datasets.py
│  ├─ model.xlsx
│  ├─ __init__.py
│  └─ __pycache__
│     └─ __init__.cpython-38.pyc
├─ modules # MetaModel 
│  ├─ loss.py
│  ├─ model.py
│  └─ trainer_main.py
├─ README.md
├─ requirements.txt
├─ run.py
├─ run.sh
└─ utils.py

```



## Citation
If you find the code useful, please cite our paper. 
```bibtex
Pending.
```

