import os.path

import torch
import numpy as np
import random
from modules.trainer_main import Trainer_main
import argparse
import time
import logging
from collections import defaultdict


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


def horizon_averages(results_dict):
    # Initialize a dictionary to store all values for each horizon
    horizon_values = {}

    # Traverse all datasets
    for dataset, horizons in results_dict.items():
        for horizon, value in horizons.items():
            if horizon not in horizon_values:
                horizon_values[horizon] = []
            horizon_values[horizon].append(value)

    # Calculate the average value for each horizon
    horizon_averages = {}
    for horizon, values in horizon_values.items():
        horizon_averages[horizon] = sum(values) / len(values)

    return horizon_averages


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multivariate Time Series Forecasting')

    parser.add_argument('--seq_len', type=int, default=96, help='Historical window size')
    parser.add_argument('--pred_len', type=int, default=96, help='Prediction length')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    parser.add_argument('--num_heads', type=int, default=8, help='Transformer heads')
    parser.add_argument('--num_layers', type=int, default=8, help='Transformer layers')
    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--in_channels', type=int, default=1, help='Channels')

    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--meta_dim', type=int, default=23, help='Meta dimension')
    parser.add_argument('--meta_outdim', type=int, default=64, help='Meta output dimension')
    parser.add_argument('--input_dim', type=int, default=512, help='Input dimension for meta')
    parser.add_argument('--d_meta_dim', type=int, default=30, help='Dimension for meta input')
    parser.add_argument('--zoo_size', type=int, default=8, help='Size of zoo for meta input')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for meta input')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--train_epochs', type=int, default=80, help='Number of training epochs')

    parser.add_argument('--mse_weight', type=float, default=0.7, help='MSE weight')
    parser.add_argument('--rank_weight', type=float, default=1.0, help='Rank weight')

    parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping')

    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--num_experts', type=int, default=4, help='Number of experts')
    parser.add_argument('--topo_dim', type=int, default=128, help='Topological dimension')
    parser.add_argument('--func_dim', type=int, default=96, help='Functional dimension')

    parser.add_argument('--setting', type=str, default='full', help="Few-shot or full-shot")
    parser.add_argument('--horizon_setting', type=str, default='all', choices=['all', '96', '192', '336', '720'], help="Horizon settings")

    parser.add_argument('--is_training', default=True, help='Enable training mode')
    parser.add_argument('--save_path', type=str, default='./log/inner0.001_meta0.005_mse0.7_epoch80_seed1_full_all_k16_q16')

    # MAML specific parameters
    parser.add_argument('--meta_batch_size', type=int, default=4, help='Number of tasks sampled per epoch')
    parser.add_argument('--k_shot', type=int, default=16, help='Number of support samples per task')
    parser.add_argument('--n_query', type=int, default=16, help='Number of query samples per task')
    parser.add_argument('--inner_lr', type=float, default=0.001, help='Learning rate for inner loop updates')
    parser.add_argument('--meta_lr', type=float, default=0.005, help='Learning rate for outer loop (meta-optimizer)')
    parser.add_argument('--num_inner_updates', type=int, default=1, help='Number of inner loop updates per task')

    parser.add_argument('--train_datasets', nargs='+', default=[], help="List of datasets used for training")
    parser.add_argument('--val_datasets', nargs='+', default=[], help="List of datasets used for validation")
    parser.add_argument('--test_datasets', nargs='+', default=[], help="List of datasets used for testing")

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    log_path = os.path.join(args.save_path, 'training_log.txt')
    logging.basicConfig(
        filename=log_path,
        filemode='w',  # Append mode ('w' for overwrite mode)
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    print('Args in experiment:')
    logging.info('Args in experiment:')
    print(args)
    logging.info(args)

    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fix_seed)
    torch.backends.cudnn.deterministic = True  # Ensure CUDA convolution results are deterministic
    torch.backends.cudnn.benchmark = False     # Disable optimization (which might introduce randomness)

    all_datasets = list(datasets_zoo.keys())
    tw_results = {}

    test_coverage = defaultdict(int)  # Record how many times each dataset has been selected as a test set
    for ds in all_datasets:
        test_coverage[ds] = 0
    all_test_combinations = []  # Store all test set combinations

    while min(test_coverage.values()) < 1:  # Until all datasets have been selected at least once
        # Randomly select 3 test sets (prioritize selecting uncovered datasets)
        uncovered_datasets = [ds for ds in all_datasets if test_coverage[ds] == 0]

        if len(uncovered_datasets) >= 3:
            # If there are still at least 3 uncovered datasets, prioritize them
            args.test_datasets = random.sample(uncovered_datasets, 3)
        else:
            # Otherwise, supplement with covered datasets to make up 3
            needed = 3 - len(uncovered_datasets)
            args.test_datasets = uncovered_datasets + random.sample(
                [ds for ds in all_datasets if ds not in uncovered_datasets], needed
            )

        # Update coverage count
        for ds in args.test_datasets:
            test_coverage[ds] += 1

        # Remaining datasets (14 - 3 = 11)
        remaining = [ds for ds in all_datasets if ds not in args.test_datasets]
        args.train_val_datasets = remaining

        print("---------------------------------------------------")
        logging.info("---------------------------------------------------")

        print(f"Train_val_datasets: {args.train_val_datasets}")
        print(f"Test datasets: {args.test_datasets}")

        logging.info(f"Train_val_datasets: {args.train_val_datasets}")
        logging.info(f"Test datasets: {args.test_datasets}")
        time_start = time.time()
        Exp = Trainer_main
        if args.is_training:
            # Meta-learning
            exp = Exp(args)  # Set experiments
            time_before = time.time()
            exp.meta_train()
            tmp_results = exp.meta_test()

            time_now = time.time()
            print(f"time:{time_now-time_before}")
            logging.info(f"time:{time_now-time_before}")

            for dataset_name in tmp_results.keys():
                if dataset_name not in tmp_results.keys():
                    tw_results[dataset_name] = {}
                tw_results[dataset_name] = tmp_results[dataset_name]
            print(tmp_results)
            logging.info(tmp_results)

            del exp
            torch.cuda.empty_cache()
        else:
            exp = Exp(args)  # Set experiments
            time_before = time.time()
            tmp_results = exp.meta_test()

            time_now = time.time()
            print(f"time:{time_now - time_before}")
            logging.info(f"time:{time_now - time_before}")

            for dataset_name in tmp_results.keys():
                if dataset_name not in tmp_results.keys():
                    tw_results[dataset_name] = {}
                tw_results[dataset_name] = tmp_results[dataset_name]
            print(tmp_results)
            logging.info(tmp_results)

            del exp
            torch.cuda.empty_cache()
        # Check if all datasets have been selected at least once
        if all(v >= 1 for v in test_coverage.values()):
            break

    print("final_results==================================")
    logging.info("final_results==================================")
    print(tw_results)
    logging.info(tw_results)

    horizon_mean = horizon_averages(tw_results)
    logging.info(horizon_mean)
    print(horizon_mean)

    tw_ = 0
    for horizon in horizon_mean.keys():
        tw_ += horizon_mean[horizon]

    tw_ /= len(horizon_mean)

    print(f"Average all tw {tw_}")
    logging.info(f"Average all tw {tw_}")