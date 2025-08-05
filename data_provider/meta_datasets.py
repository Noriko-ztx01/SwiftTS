import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import os
import torch
import random
import torch.nn.functional as F
from collections import defaultdict


class MetaDatasets(Dataset):
    """
    Dataset class for meta-testing or inference.
    Each sample contains time series features, metadata, and model performance information.
    """
    def __init__(self, data_samples):
        self.samples = data_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'data_features': torch.from_numpy(sample['data_features']).float(),
            'd_meta_embed': torch.tensor(sample['d_meta_embed']).float(),
            'model_rank': torch.from_numpy(sample['model_rank']).float(),
            'm_meta_embed': torch.tensor(sample['m_meta_embed']).float(),
            'm_topo': torch.from_numpy(sample['m_topo']).float(),
            'm_func': torch.from_numpy(sample['m_func']).float(),
            'm_means': sample['m_means'],
            'm_stds': sample['m_stds'],
            'horizon': torch.tensor(sample['horizon']).float(),
            'dataset': sample['dataset'],
        }


class MetaTrainDatasets(Dataset):
    """
    Dataset class for meta-training.
    Supports sampling tasks (support and query sets) for meta-learning.
    """
    def __init__(self, data_samples):
        self.samples = data_samples
        # Group indices by horizon; each horizon contains data from multiple datasets
        self.horizon_to_indices = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            horizon = sample['horizon']
            self.horizon_to_indices[horizon].append(idx)
        self.fix_seed = 1  # Fixed seed for reproducibility

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'data_features': torch.from_numpy(sample['data_features']).float(),
            'd_meta_embed': torch.tensor(sample['d_meta_embed']).float(),
            'model_rank': torch.from_numpy(sample['model_rank']).float(),
            'm_meta_embed': torch.tensor(sample['m_meta_embed']).float(),
            'm_topo': torch.from_numpy(sample['m_topo']).float(),
            'm_func': torch.from_numpy(sample['m_func']).float(),
            'm_means': sample['m_means'],
            'm_stds': sample['m_stds'],
            'horizon': torch.tensor(sample['horizon']).float(),
            'dataset': sample['dataset'],
        }

    def sample_task(self, k_shot=5, n_query=5):
        """
        Sample a meta-learning task:
        - Randomly select k_shot samples from all data as the support set
        - Randomly select n_query samples from all data as the query set

        Args:
            k_shot (int): Number of support samples per task
            n_query (int): Number of query samples per task

        Returns:
            tuple: (support_set, query_set), each is a list of sampled items
        """
        # Set all random seeds for reproducibility
        self._set_seeds()

        total_samples = len(self.samples)
        if k_shot + n_query > total_samples:
            raise ValueError(f"Requested {k_shot + n_query} samples but only {total_samples} available")

        # Randomly sample k_shot + n_query indices
        shuffled_indices = random.sample(range(total_samples), k_shot + n_query)

        # Split into support and query sets
        support_indices = shuffled_indices[:k_shot]
        query_indices = shuffled_indices[k_shot:]

        # Retrieve support and query sets
        support_set = [self.__getitem__(i) for i in support_indices]
        query_set = [self.__getitem__(i) for i in query_indices]

        return support_set, query_set

    def _set_seeds(self):
        """
        Set all relevant random seeds for reproducibility.
        """
        random.seed(self.fix_seed)
        os.environ['PYTHONHASHSEED'] = str(self.fix_seed)
        np.random.seed(self.fix_seed)
        torch.manual_seed(self.fix_seed)
        torch.cuda.manual_seed(self.fix_seed)
        torch.cuda.manual_seed_all(self.fix_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_datasets(datasets_zoo, train_val_datasets, test_datasets, split_ratio=7/8, setting="few", horizon_setting="all"):
    """
    Load .pkl files, combine train+val datasets, and split into train and validation sets.

    Args:
        datasets_zoo (dict): Mapping of dataset names to IDs
        train_val_datasets (list): List of dataset names for training and validation
        test_datasets (list): List of dataset names for testing
        split_ratio (float): Proportion of data for training (e.g., 7/8 means train:val = 7:1)
        setting (str): Data setting (e.g., 'few' for few-shot)
        horizon_setting (str): Horizon setting (e.g., 'all' for all horizons)

    Returns:
        tuple: (train_data, val_data, test_data) lists of samples
    """
    # Set fixed seeds for reproducibility
    fix_seed = 1
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    all_samples = []
    for dataset in datasets_zoo.keys():
        file_path = f"../datasets/{setting}_{horizon_setting}/{dataset}_{setting}_{horizon_setting}.pkl"
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            all_samples.extend(data)

    # Separate train+val and test data
    train_val_data = [s for s in all_samples if s['dataset'] in train_val_datasets]
    test_data = [s for s in all_samples if s['dataset'] in test_datasets]

    # Shuffle and split train+val into train and validation sets
    random.shuffle(train_val_data)
    split_idx = int(len(train_val_data) * split_ratio)
    train_data = train_val_data[:split_idx]
    val_data = train_val_data[split_idx:]

    return train_data, val_data, test_data


def create_metadatasets(batch_size=32, train_val_datasets=None, test_datasets=None, split_ratio=7/8, setting="few", horizon_setting="all"):
    """
    Create meta-datasets and data loaders for training, validation, and testing.

    Args:
        batch_size (int): Batch size for data loaders
        train_val_datasets (list): Datasets for training and validation
        test_datasets (list): Datasets for testing
        split_ratio (float): Train/val split ratio
        setting (str): Data setting (e.g., 'few')
        horizon_setting (str): Horizon setting (e.g., 'all')

    Returns:
        tuple: (train_dataset, val_dataset, test_loader)
    """
    datasets_zoo = {
        "ETTh1": 0,
        "ETTh2": 1,
        "ETTm1": 2,
        "ETTm2": 3,
        "Electricity": 4,
        "Traffic": 5,
        "Solar": 6,
        "Weather": 7,
        "Exchange": 8,
        "ZafNoo": 9,
        'Wind': 10,
        'CzeLan': 11,
        'PEMS08': 12,
        'AQShunyi': 13
    }

    # Load and split data
    train_data, val_data, test_data = load_datasets(
        datasets_zoo, train_val_datasets, test_datasets, split_ratio, setting, horizon_setting
    )

    # Create Dataset instances
    train_dataset = MetaTrainDatasets(train_data)
    val_dataset = MetaTrainDatasets(val_data)
    test_dataset = MetaDatasets(test_data)

    # Create DataLoader for test (train and val use custom sampling via sample_task)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset, test_loader