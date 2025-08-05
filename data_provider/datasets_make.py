#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
import random
import torch


trans_full_mse = {
    'ETTh1': {
        96: {'Chronos': 0.388, 'TimesFM': 0.373, 'Moment': 0.383, 'UniTS': 0.399,
             'Moirai': 0.394, 'TinyTimeMixer': 0.361, 'RoseModel': 0.354, 'TimerModel': 0.416},
        192: {'Chronos': 0.44, 'TimesFM': 0.418, 'Moment': 0.415, 'UniTS': 0.441,
              'Moirai': 0.43, 'TinyTimeMixer': 0.393, 'RoseModel': 0.389, 'TimerModel': 0.557},
        336: {'Chronos': 0.477, 'TimesFM': 0.457, 'Moment': 0.425, 'UniTS': 0.503,
              'Moirai': 0.45, 'TinyTimeMixer': 0.411, 'RoseModel': 0.406, 'TimerModel': 0.502},
        720: {'Chronos': 0.475, 'TimesFM': 0.458, 'Moment': 0.447, 'UniTS': 0.468,
              'Moirai': 0.457, 'TinyTimeMixer': 0.426, 'RoseModel': 0.413, 'TimerModel': 0.525}},
    'ETTh2': {
        96: {'Chronos': 0.292, 'TimesFM': 0.288, 'Moment': 0.287, 'UniTS': 0.311,
             'Moirai': 0.285, 'TinyTimeMixer': 0.27, 'RoseModel': 0.265, 'TimerModel': 0.305},
        192: {'Chronos': 0.362, 'TimesFM': 0.371, 'Moment': 0.35, 'UniTS': 0.47,
              'Moirai': 0.352, 'TinyTimeMixer': 0.338, 'RoseModel': 0.328, 'TimerModel': 0.394},
        336: {'Chronos': 0.404, 'TimesFM': 0.418, 'Moment': 0.367, 'UniTS': 0.429,
              'Moirai': 0.384, 'TinyTimeMixer': 0.367, 'RoseModel': 0.353, 'TimerModel': 0.414},
        720: {'Chronos': 0.412, 'TimesFM': 0.441, 'Moment': 0.404, 'UniTS': 0.424,
              'Moirai': 0.419, 'TinyTimeMixer': 0.384, 'RoseModel': 0.376, 'TimerModel': 0.521}},
    'ETTm1': {
        96: {'Chronos': 0.339, 'TimesFM': 0.313, 'Moment': 0.287, 'UniTS': 0.321,
             'Moirai': 0.464, 'TinyTimeMixer': 0.285, 'RoseModel': 0.275, 'TimerModel': 0.344},
        192: {'Chronos': 0.392, 'TimesFM': 0.353, 'Moment': 0.326, 'UniTS': 0.373,
              'Moirai': 0.488, 'TinyTimeMixer': 0.325, 'RoseModel': 0.324, 'TimerModel': 0.447},
        336: {'Chronos': 0.44, 'TimesFM': 1.177, 'Moment': 0.353, 'UniTS': 0.388,
              'Moirai': 0.52, 'TinyTimeMixer': 0.357, 'RoseModel': 0.354, 'TimerModel': 0.457},
        720: {'Chronos': 0.53, 'TimesFM': 1.095, 'Moment': 0.408, 'UniTS': 0.452,
              'Moirai': 0.598, 'TinyTimeMixer': 0.413, 'RoseModel': 0.411, 'TimerModel': 1.444}},
    'ETTm2': {
        96: {'Chronos': 0.181, 'TimesFM': 0.172, 'Moment': 0.17, 'UniTS': 0.198,
             'Moirai': 0.224, 'TinyTimeMixer': 0.165, 'RoseModel': 0.157, 'TimerModel': 0.188},
        192: {'Chronos': 0.253, 'TimesFM': 0.234, 'Moment': 0.23, 'UniTS': 0.252,
              'Moirai': 0.308, 'TinyTimeMixer': 0.225, 'RoseModel': 0.213, 'TimerModel': 0.281},
        336: {'Chronos': 0.318, 'TimesFM': 0.357, 'Moment': 0.283, 'UniTS': 0.334,
              'Moirai': 0.369, 'TinyTimeMixer': 0.275, 'RoseModel': 0.266, 'TimerModel': 0.328},
        720: {'Chronos': 0.417, 'TimesFM': 0.454, 'Moment': 0.375, 'UniTS': 0.468,
              'Moirai': 0.46, 'TinyTimeMixer': 0.367, 'RoseModel': 0.347, 'TimerModel': 0.493}},
    'Electricity': {
        96: {'Chronos': 0.133, 'TimesFM': 0.142, 'Moment': 0.148, 'UniTS': 0.133,
             'Moirai': 0.17, 'TinyTimeMixer': 0.132, 'RoseModel': 0.125, 'TimerModel': 0.136},
        192: {'Chronos': 0.152, 'TimesFM': 0.163, 'Moment': 0.165, 'UniTS': 0.153,
              'Moirai': 0.186, 'TinyTimeMixer': 0.149, 'RoseModel': 0.142, 'TimerModel': 0.169},
        336: {'Chronos': 0.171, 'TimesFM': 0.332, 'Moment': 0.182, 'UniTS': 0.175,
              'Moirai': 0.205, 'TinyTimeMixer': 0.27, 'RoseModel': 0.162, 'TimerModel': 0.196},
        720: {'Chronos': 0.201, 'TimesFM': 0.364, 'Moment': 0.223, 'UniTS': 0.204,
              'Moirai': 0.247, 'TinyTimeMixer': 0.297, 'RoseModel': 0.191, 'TimerModel': 0.364}},
    'Traffic': {
        96: {'Chronos': 0.385, 'TimesFM': 0.419, 'Moment': 0.383, 'UniTS': 0.377,
             'Moirai': 0.358, 'TinyTimeMixer': 0.379, 'RoseModel': 0.354, 'TimerModel': 0.362},
        192: {'Chronos': 0.411, 'TimesFM': 0.45, 'Moment': 0.397, 'UniTS': 0.387,
              'Moirai': 0.372, 'TinyTimeMixer': 0.396, 'RoseModel': 0.377, 'TimerModel': 0.396},
        336: {'Chronos': 0.521, 'TimesFM': 0.939, 'Moment': 0.407, 'UniTS': 0.395,
              'Moirai': 0.38, 'TinyTimeMixer': 0.945, 'RoseModel': 0.396, 'TimerModel': 0.427},
        720: {'Chronos': 0.623, 'TimesFM': 0.957, 'Moment': 0.443, 'UniTS': 0.436,
              'Moirai': 0.412, 'TinyTimeMixer': 0.952, 'RoseModel': 0.434, 'TimerModel': 0.97}},
    'Solar': {
        96: {'Chronos': 0.43, 'TimesFM': 0.174, 'Moment': 0.172, 'UniTS': 0.163,
             'Moirai': 0.877, 'TinyTimeMixer': 0.174, 'RoseModel': 0.17, 'TimerModel': 0.183},
        192: {'Chronos': 0.396, 'TimesFM': 0.198, 'Moment': 0.187, 'UniTS': 0.176,
              'Moirai': 0.928, 'TinyTimeMixer': 0.181, 'RoseModel': 0.204, 'TimerModel': 0.225},
        336: {'Chronos': 0.409, 'TimesFM': 1.53, 'Moment': 0.196, 'UniTS': 0.184,
              'Moirai': 0.956, 'TinyTimeMixer': 0.189, 'RoseModel': 1.616, 'TimerModel': 0.244},
        720: {'Chronos': 0.453, 'TimesFM': 1.322, 'Moment': 0.206, 'UniTS': 0.196,
              'Moirai': 1.016, 'TinyTimeMixer': 0.2, 'RoseModel': 0.215, 'TimerModel': 0.355}},
    'Weather': {
        96: {'Chronos': 0.183, 'TimesFM': 0.161, 'Moment': 0.152, 'UniTS': 0.147,
             'Moirai': 0.206, 'TinyTimeMixer': 0.149, 'RoseModel': 0.145, 'TimerModel': 0.164},
        192: {'Chronos': 0.227, 'TimesFM': 0.207, 'Moment': 0.196, 'UniTS': 0.191,
              'Moirai': 0.278, 'TinyTimeMixer': 0.199, 'RoseModel': 0.183, 'TimerModel': 0.243},
        336: {'Chronos': 0.286, 'TimesFM': 0.311, 'Moment': 0.245, 'UniTS': 0.243,
              'Moirai': 0.335, 'TinyTimeMixer': 0.256, 'RoseModel': 0.232, 'TimerModel': 0.321},
        720: {'Chronos': 0.368, 'TimesFM': 0.37, 'Moment': 0.316, 'UniTS': 0.317,
              'Moirai': 0.413, 'TinyTimeMixer': 0.34, 'RoseModel': 0.309, 'TimerModel': 0.349}},
    'Exchange': {
        96: {'Chronos': 0.093, 'TimesFM': 0.086, 'Moment': 0.085, 'UniTS': 0.444,
             'Moirai': 0.096, 'TinyTimeMixer': 0.113, 'RoseModel': 0.086, 'TimerModel': 0.104},
        192: {'Chronos': 0.199, 'TimesFM': 0.193, 'Moment': 0.178, 'UniTS': 0.507,
              'Moirai': 0.2, 'TinyTimeMixer': 0.223, 'RoseModel': 0.178, 'TimerModel': 0.221},
        336: {'Chronos': 0.37, 'TimesFM': 0.354, 'Moment': 0.333, 'UniTS': 0.489,
              'Moirai': 0.381, 'TinyTimeMixer': 0.439, 'RoseModel': 0.341, 'TimerModel': 0.382},
        720: {'Chronos': 0.856, 'TimesFM': 0.988, 'Moment': 0.851, 'UniTS': 0.997,
              'Moirai': 1.133, 'TinyTimeMixer': 1.185, 'RoseModel': 0.947, 'TimerModel': 0.965}},
    'ZafNoo': {
        96: {'Chronos': 0.463, 'TimesFM': 0.457, 'Moment': 0.43, 'UniTS': 0.444,
             'Moirai': 0.439, 'TinyTimeMixer': 0.426, 'RoseModel': 0.431, 'TimerModel': 0.47},
        192: {'Chronos': 0.524, 'TimesFM': 0.576, 'Moment': 0.486, 'UniTS': 0.507,
              'Moirai': 0.501, 'TinyTimeMixer': 0.479, 'RoseModel': 0.487, 'TimerModel': 0.548},
        336: {'Chronos': 0.575, 'TimesFM': 0.65, 'Moment': 0.53, 'UniTS': 0.563,
              'Moirai': 0.551, 'TinyTimeMixer': 0.523, 'RoseModel': 0.538, 'TimerModel': 0.588},
        720: {'Chronos': 0.684, 'TimesFM': 0.748, 'Moment': 0.585, 'UniTS': 0.602,
              'Moirai': 0.616, 'TinyTimeMixer': 0.583, 'RoseModel': 0.578, 'TimerModel': 0.637}},
    'Wind': {
        96: {'Chronos': 1.177, 'TimesFM': 0.913, 'Moment': 0.915, 'UniTS': 0.949,
             'Moirai': 0.957, 'TinyTimeMixer': 0.889, 'RoseModel': 0.904, 'TimerModel': 1.087},
        192: {'Chronos': 1.391, 'TimesFM': 1.098, 'Moment': 1.101, 'UniTS': 1.151,
              'Moirai': 1.164, 'TinyTimeMixer': 1.056, 'RoseModel': 1.086, 'TimerModel': 1.341},
        336: {'Chronos': 1.54, 'TimesFM': 1.326, 'Moment': 1.231, 'UniTS': 1.329,
              'Moirai': 1.333, 'TinyTimeMixer': 1.189, 'RoseModel': 1.238, 'TimerModel': 1.514},
        720: {'Chronos': 1.685, 'TimesFM': 1.437, 'Moment': 1.303, 'UniTS': 1.545,
              'Moirai': 1.466, 'TinyTimeMixer': 1.271, 'RoseModel': 1.33, 'TimerModel': 1.751}},
    'CzeLan': {
        96: {'Chronos': 0.505, 'TimesFM': 0.198, 'Moment': 0.171, 'UniTS': 0.196,
             'Moirai': 0.611, 'TinyTimeMixer': 0.162, 'RoseModel': 0.164, 'TimerModel': 0.224},
        192: {'Chronos': 0.565, 'TimesFM': 0.244, 'Moment': 0.201, 'UniTS': 0.226,
              'Moirai': 0.623, 'TinyTimeMixer': 0.192, 'RoseModel': 0.198, 'TimerModel': 1.198},
        336: {'Chronos': 0.669, 'TimesFM': 1.232, 'Moment': 0.225, 'UniTS': 0.25,
              'Moirai': 0.654, 'TinyTimeMixer': 0.217, 'RoseModel': 0.221, 'TimerModel': 0.75},
        720: {'Chronos': 0.838, 'TimesFM': 1.214, 'Moment': 0.264, 'UniTS': 0.323,
              'Moirai': 0.702, 'TinyTimeMixer': 0.253, 'RoseModel': 0.253, 'TimerModel': 0.848}},
    'PEMS08': {
        96: {'Chronos': 0.804, 'TimesFM': 0.167, 'Moment': 0.261, 'UniTS': 0.519,
             'Moirai': 0.144, 'TinyTimeMixer': 0.177, 'RoseModel': 0.199, 'TimerModel': 0.194},
        192: {'Chronos': 1.264, 'TimesFM': 0.267, 'Moment': 0.335, 'UniTS': 0.654,
              'Moirai': 0.211, 'TinyTimeMixer': 0.268, 'RoseModel': 0.391, 'TimerModel': 0.359},
        336: {'Chronos': 1.317, 'TimesFM': 1.285, 'Moment': 0.365, 'UniTS': 0.599,
              'Moirai': 0.276, 'TinyTimeMixer': 1.206, 'RoseModel': 1.441, 'TimerModel': 0.385},
        720: {'Chronos': 1.521, 'TimesFM': 1.111, 'Moment': 0.381, 'UniTS': 0.66,
              'Moirai': 0.333, 'TinyTimeMixer': 1.097, 'RoseModel': 1.351, 'TimerModel': 2.235}},
    'AQShunyi': {
        96: {'Chronos': 0.728, 'TimesFM': 0.662, 'Moment': 0.66, 'UniTS': 0.739,
             'Moirai': 0.621, 'TinyTimeMixer': 0.64, 'RoseModel': 0.632, 'TimerModel': 0.814},
        192: {'Chronos': 0.802, 'TimesFM': 0.746, 'Moment': 0.707, 'UniTS': 0.784,
              'Moirai': 0.665, 'TinyTimeMixer': 0.683, 'RoseModel': 0.677, 'TimerModel': 0.882},
        336: {'Chronos': 0.843, 'TimesFM': 0.795, 'Moment': 0.727, 'UniTS': 0.829,
              'Moirai': 0.697, 'TinyTimeMixer': 0.706, 'RoseModel': 0.706, 'TimerModel': 0.89},
        720: {'Chronos': 0.897, 'TimesFM': 0.82, 'Moment': 0.782, 'UniTS': 0.857,
              'Moirai': 0.74, 'TinyTimeMixer': 0.763, 'RoseModel': 0.77, 'TimerModel': 0.953}}
}

model_zoo = {
    "Chronos": 0,
    "TimesFM": 1,
    "Moment": 2,
    "Moirai": 3,
    "UniTS": 4,
    "TinyTimeMixer": 5,
    "RoseModel": 6,
    "TimerModel": 7
}



def data_meta_embedding(dataset_name):
    """
    Encode dataset metadata into a feature vector.

    Args:
        dataset_name (str): Name of the dataset for which to generate the embedding.

    Returns:
        pd.DataFrame: A DataFrame containing encoded metadata features.
    """
    data_meta_path = 'dataset.xlsx'
    data_df = pd.read_excel(data_meta_path)
    # Temporarily do not select Descriptions
    dataset_selected_col = ['Dataset', 'Variables', 'Timestamps', 'Domain',
                            'Frequency', 'Seasonality', 'Trend', 'Stationarity',
                            'Transition', 'Shifting', 'Correlation', 'N-Gau']
    data_df = data_df[dataset_selected_col]  # Only 'Dataset' needs one-hot encoding
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # Use sparse_output parameter
    columns_to_encode = ['Dataset']
    for column in columns_to_encode:
        column_data = data_df[[column]]
        encoded_data = ohe.fit_transform(column_data)
        encoded_columns = ohe.get_feature_names_out([column])
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=data_df.index)
        data_df = pd.concat([data_df, encoded_df], axis=1)

    domain_mlb = MultiLabelBinarizer()
    data_df['Domain'] = data_df['Domain'].apply(lambda x: [x])  # Convert single label to list format
    domain_encoded = domain_mlb.fit_transform(data_df['Domain'])
    domain_df = pd.DataFrame(domain_encoded, columns=domain_mlb.classes_, index=data_df.index)
    domain_df.columns = [f'Domain_{col}' for col in domain_df.columns]  # Rename columns

    data_df = pd.concat([data_df, domain_df], axis=1)

    data_info = data_df.loc[data_df['Dataset'] == dataset_name].copy()
    data_info = data_info.drop(columns=['Dataset', 'Domain'])
    return data_info


def model_meta_embedding(model_name):
    """
    Encode model metadata into a feature vector.

    Args:
        model_name (str): Name of the model for which to generate the embedding.

    Returns:
        pd.DataFrame: A DataFrame containing encoded metadata features.
    """
    model_meta_path = 'model.xlsx'  # Replace with your file path
    model_df = pd.read_excel(model_meta_path)
    model_selected_col = ['Model', 'architecture', 'parameters', 'gmacs', 'Pre-trained data']
    model_df = model_df[model_selected_col]
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # Use sparse_output parameter
    columns_to_encode = ['Model', 'architecture']
    for column in columns_to_encode:
        column_data = model_df[[column]]
        encoded_data = ohe.fit_transform(column_data)
        encoded_columns = ohe.get_feature_names_out([column])
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=model_df.index)
        model_df = pd.concat([model_df, encoded_df], axis=1)

    model_df['Pre-trained data'] = model_df['Pre-trained data'].apply(lambda x: x.split(','))

    pretrained_mlb = MultiLabelBinarizer()
    pretrained_encoded = pretrained_mlb.fit_transform(model_df['Pre-trained data'])

    pretrained_df = pd.DataFrame(pretrained_encoded, columns=pretrained_mlb.classes_, index=model_df.index)
    pretrained_df.columns = [f'Pre-trained_{col}' for col in pretrained_df.columns]  # Rename columns

    model_df = pd.concat([model_df, pretrained_df], axis=1)

    model_info = model_df.loc[model_df['Model'] == model_name].copy()
    model_info = model_info.drop(columns=['Model', 'architecture', 'Pre-trained data'])
    return model_info


def all_meta_embedding():
    """
    Create meta-embeddings for all models.

    Returns:
        tuple: A tuple containing model embeddings, their means and standard deviations.
    """
    models_meta = []
    for model in model_zoo.keys():
        m_meta_emb = model_meta_embedding(model)
        m_meta_emb = m_meta_emb.to_numpy()
        m_meta_emb = torch.from_numpy(m_meta_emb)
        models_meta.append(m_meta_emb)
    models_meta = torch.cat([x for x in models_meta], dim=0)

    cols_to_normalize = [0, 1]
    m_means = models_meta[:, cols_to_normalize].mean(dim=0)  # Calculate mean across columns
    m_stds = models_meta[:, cols_to_normalize].std(dim=0)  # Calculate std across columns
    m_stds = m_stds + 1e-8  # Avoid division by zero
    models_meta[:, cols_to_normalize] = (models_meta[:, cols_to_normalize] - m_means) / m_stds
    return models_meta, m_means, m_stds


def all_model_embedding():
    """
    Load topology and function embeddings for all models.

    Returns:
        tuple: A tuple containing topology and function embeddings.
    """
    topo = []
    func = []
    for model_name in model_zoo.keys():
        topo_emb_path = f'../datasets/embedding/{model_name}_topo_emb.npy'
        func_emb_path = f'../datasets/embedding/{model_name}_func_emb.npy'
        topo_emb = np.load(topo_emb_path)
        func_emb = np.load(func_emb_path)

        if func_emb.shape[1] == 192:
            func_emb = func_emb[:, :96, :]
        func_emb = np.mean(func_emb, axis=0)
        func_emb = np.expand_dims(np.mean(func_emb, axis=1), axis=0)
        topo.append(topo_emb)
        func.append(func_emb)

    topo = np.concatenate(topo, axis=0)
    func = np.concatenate(func, axis=0)

    return topo, func


# Main code
if __name__ == "__main__":
    setting = "full"
    horizon_setting = "all"

    horizon_list = []
    if horizon_setting == "all":
        horizon_list = [96, 192, 336, 720]
    elif horizon_setting in ["96", "192", "336", "720"]:
        horizon_list = [int(horizon_setting)]
    else:
        raise ValueError("Invalid horizon value")

    fix_seed = 19
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    models_meta, m_means, m_stds = all_meta_embedding()
    topo, func = all_model_embedding()  # [8, 128], [8, 96]

    data_hub = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Electricity', 'Traffic', 'Solar',
                'Weather', 'Exchange', 'ZafNoo', 'Wind', 'CzeLan', 'PEMS08', 'AQShunyi']

    datas_meta = []
    for data in data_hub:
        d_meta_emb = data_meta_embedding(data)
        d_meta_emb = d_meta_emb.to_numpy()
        d_meta_emb = torch.from_numpy(d_meta_emb)
        datas_meta.append(d_meta_emb)
    datas_meta = torch.cat([x for x in datas_meta], dim=0)
    cols_to_normalize = list(range(10))  # Normalize the first 10 columns
    d_means = datas_meta[:, cols_to_normalize].mean(dim=0)
    d_stds = datas_meta[:, cols_to_normalize].std(dim=0)
    d_stds = d_stds + 1e-8
    datas_meta[:, cols_to_normalize] = (datas_meta[:, cols_to_normalize] - d_means) / d_stds

    data_id = 0
    for data in data_hub:
        infea_path = os.path.join('../results_f_all', f'{data}_feats.npy')
        in_features = np.load(infea_path)
        in_features = torch.from_numpy(in_features)

        all_data = []

        batch_size = 64
        num_batches = int(in_features.shape[0] // batch_size)

        sampled_indices = set()

        assert num_batches * batch_size <= in_features.shape[0], "Sampling total exceeds dataset size!"

        for i in range(num_batches):
            while True:
                batch_indices = torch.randint(0, in_features.shape[0], (batch_size,))
                if all(idx not in sampled_indices for idx in batch_indices):
                    break

            sampled_indices.update(batch_indices.tolist())
            sampled_tensor = in_features[batch_indices]
            sampled_tensor = torch.mean(sampled_tensor, dim=0)
            sampled_tensor = torch.mean(sampled_tensor, dim=0)

            for horizon in horizon_list:
                ground_rank = trans_full_mse[data][horizon]
                model_rank = [ground_rank[model] for model in model_zoo.keys()]

                sample = {
                    "data_features": sampled_tensor.cpu().numpy(),
                    "d_meta_embed": datas_meta[data_id].cpu().numpy(),
                    "model_rank": np.array(model_rank),
                    "m_meta_embed": models_meta.cpu().numpy(),
                    "m_topo": topo,
                    "m_func": func,
                    "m_means": m_means,
                    "m_stds": m_stds,
                    "horizon": horizon,
                    "dataset": data
                }
                all_data.append(sample)

        save_dir = f"../datasets/{setting}_{horizon_setting}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = f"{save_dir}/{data}_{setting}_{horizon_setting}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(all_data, f)
        print(f"Data saved to {save_path} with {len(all_data)} samples.")

        data_id += 1