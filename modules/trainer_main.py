import collections
import logging
import os
import copy
import numpy as np
import torch
from tqdm import tqdm
import random

from data_provider.meta_datasets import create_metadatasets
from modules.model import MetaModel
from modules.loss import w_kendall_metric, Top1CE, copelands_aggregation


def horizon_averages(results_dict):
    """
    Calculate the average values for each horizon across all datasets.

    Args:
        results_dict (dict): A nested dictionary containing dataset names as keys,
                             horizons as subkeys, and their corresponding values.

    Returns:
        dict: A dictionary where each key is a horizon and its value is the average
              of that horizon's values across all datasets.
    """

    # Initialize a dictionary to store all values for each horizon
    horizon_values = {}

    # Iterate over all datasets
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



class Trainer_main():
    """
    Main trainer class for training and evaluating a meta-learning model.
    Handles device setup, model initialization, data loading, and training configuration.
    """

    def __init__(self, args):
        """
        Initialize the trainer with command-line arguments.

        Args:
            args (argparse.Namespace): Parsed command-line arguments containing model and training configurations.
        """
        self.args = args
        self.device = self._acquire_device()  # Set up the device (CPU or GPU)
        self.best_model = None  # Placeholder for storing the best performing model
        self._get_data()         # Load training, validation, and test datasets
        self._build_model()      # Build and initialize the model

        # Set random seed for reproducibility
        fix_seed = args.seed
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fix_seed)  # For multi-GPU setups
        # Ensure deterministic behavior in CUDA operations
        torch.backends.cudnn.deterministic = True  # Makes CUDA convolution results reproducible
        torch.backends.cudnn.benchmark = False     # Disables cuDNN auto-tuner to avoid randomness

    def _build_model(self):
        """
        Build the main model using parameters from self.args.
        Initializes the MetaModel with specified hyperparameters and moves it to the appropriate device.
        Also sets up the optimizer and loss functions.
        """
        # Directly use arguments from args instead of ForecastingConfig
        self.model = MetaModel(
            d_model=self.args.d_model,
            embed_dim=self.args.embed_dim,
            num_heads=self.args.num_heads,
            meta_dim=self.args.meta_dim,
            meta_outdim=self.args.meta_outdim,
            topo_dim=self.args.topo_dim,
            func_dim=self.args.func_dim,
            d_meta_dim=self.args.d_meta_dim,
            num_experts=self.args.num_experts
        ).to(self.device)

        # Collect all model parameters for optimization
        self.parameters = [*self.model.parameters()]

        # Use Adam optimizer with meta-learning rate
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.args.meta_lr)

        # Define loss functions
        self.criterion_mse = torch.nn.MSELoss()  # Mean Squared Error loss
        self.criterion = Top1CE()                # Custom Top-1 Cross Entropy loss (assumed custom class)

        # Initialize early stopping mechanism
        self.early_stopping = EarlyStopping(patience=self.args.patience)

    def _acquire_device(self):
        """
        Determine and return the appropriate device for training (GPU if available, otherwise CPU).

        Returns:
            torch.device: The device object (either 'cuda' or 'cpu').
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

    def _get_data(self):
        """
        Load meta-training, meta-validation, and test datasets using the create_metadatasets function.
        Prints and logs dataset sizes for monitoring.
        """
        print('==> loading data loaders ... ')
        logging.info('==> loading data loaders ... ')

        # Create meta-datasets for training, validation, and testing
        self.metatrain_datasets, self.metaval_datasets, self.test_loader = create_metadatasets(
            batch_size=self.args.batch_size,
            train_val_datasets=self.args.train_val_datasets,
            test_datasets=self.args.test_datasets,
            setting=self.args.setting,
            horizon_setting=self.args.horizon_setting
        )

        # Print dataset sizes
        print("Train dataset size:", len(self.metatrain_datasets))
        print("Validation dataset size:", len(self.metaval_datasets))
        print("Test dataset size:", len(self.test_loader.dataset))

        # Log dataset sizes
        logging.info(f"Train dataset size: {len(self.metatrain_datasets)}")
        logging.info(f"Validation dataset size: {len(self.metaval_datasets)}")
        logging.info(f"Test dataset size: {len(self.test_loader.dataset)}")

    def meta_train(self):
        """
        Train the MetaModel using a meta-learning (e.g., MAML-style) approach.
        This involves sampling tasks, performing inner-loop adaptation on support sets,
        and updating the model based on query set performance (outer loop).
        """
        if self.args.is_training:
            print('==> Training MetaModel ... ')
            logging.info('==> Training MetaModel ... ')

            # Initialize lists to track training and validation loss over epochs
            train_loss = []
            vali_loss = []
            min_loss = float('inf')  # Track the minimum validation loss

            # Set the meta-batch size based on the number of available datasets and k-shot
            self.args.meta_batch_size = len(self.metatrain_datasets) // self.args.k_shot

            # Training loop over epochs
            for epoch in range(self.args.train_epochs):
                self.model.train()  # Set model to training mode
                print('Epoch: {}'.format(epoch))
                logging.info('Epoch: {}'.format(epoch))
                epoch_loss = 0.0  # Accumulate loss for the current epoch

                # Iterate over meta-batches (each batch is a task)
                for task_idx in range(self.args.meta_batch_size):
                    # Sample a task consisting of a support set and a query set
                    support_set, query_set = self.metatrain_datasets.sample_task(
                        k_shot=self.args.k_shot,
                        n_query=self.args.n_query
                    )

                    # Inner loop: Adapt on the support set using fast weights
                    # Create a copy of the current model parameters (fast weights)
                    fast_weights = collections.OrderedDict(self.model.named_parameters())
                    inner_loss = 0.0

                    # Perform multiple inner gradient updates
                    for _ in range(self.args.num_inner_updates):
                        support_loss = 0.0
                        # Compute loss over all samples in the support set
                        for sample in support_set:
                            x = sample['data_features'].unsqueeze(0).to(self.device)
                            d_meta_emb = sample['d_meta_embed'].unsqueeze(0).to(self.device)
                            horizon = sample['horizon'].to(self.device)
                            m_meta_emb = sample['m_meta_embed'].unsqueeze(0).to(self.device)
                            m_topo_emb = sample['m_topo'].unsqueeze(0).to(self.device)
                            m_func_emb = sample['m_func'].unsqueeze(0).to(self.device)
                            y = sample['model_rank'].unsqueeze(0).to(self.device)

                            # Forward pass using fast weights
                            prediction, mean_embed, model_embed, attn_output = self.model.functional_forward(
                                x, m_meta_emb, d_meta_emb, m_topo_emb, m_func_emb, horizon, params=fast_weights
                            )

                            # Compute ranking and MSE loss
                            rank_loss = self.criterion(prediction, y)
                            mse_loss = self.criterion_mse(prediction, y)
                            tmp_loss = self.args.rank_weight * rank_loss + self.args.mse_weight * mse_loss
                            support_loss += tmp_loss

                        # Average loss over inner updates
                        inner_loss = support_loss / len(
                            support_set)  # Note: original uses num_inner_updates, but likely should be len(support_set)

                        # Compute gradients with respect to fast weights
                        grads = torch.autograd.grad(inner_loss, fast_weights.values(), create_graph=True,
                                                    allow_unused=True)

                        # Update fast weights using inner learning rate
                        fast_weights = collections.OrderedDict(
                            (name, param - self.args.inner_lr * grad)
                            for ((name, param), grad) in zip(fast_weights.items(), grads)
                        )

                    # Outer loop: Compute meta-loss on the query set using adapted fast weights
                    query_loss = 0.0
                    for sample in query_set:
                        x = sample['data_features'].unsqueeze(0).to(self.device)
                        d_meta_emb = sample['d_meta_embed'].unsqueeze(0).to(self.device)
                        horizon = sample['horizon'].to(self.device)
                        m_meta_emb = sample['m_meta_embed'].unsqueeze(0).to(self.device)
                        m_topo_emb = sample['m_topo'].unsqueeze(0).to(self.device)
                        m_func_emb = sample['m_func'].unsqueeze(0).to(self.device)
                        y = sample['model_rank'].unsqueeze(0).to(self.device)

                        # Forward pass on query set using adapted fast weights
                        prediction, _, _, _ = self.model.functional_forward(
                            x, m_meta_emb, d_meta_emb, m_topo_emb, m_func_emb, horizon, params=fast_weights
                        )

                        # Compute loss
                        rank_loss = self.criterion(prediction, y)
                        mse_loss = self.criterion_mse(prediction, y)
                        tmp_loss = self.args.rank_weight * rank_loss + self.args.mse_weight * mse_loss
                        query_loss += tmp_loss

                    total_loss = query_loss  # Meta-loss is the sum over query set

                    # Update the original model parameters (slow weights)
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

                    # Accumulate total loss for the epoch
                    epoch_loss += total_loss.item()

                # Log average training loss for the epoch
                avg_epoch_loss = epoch_loss / self.args.meta_batch_size
                print(f"Epoch {epoch}, Train Loss: {avg_epoch_loss}")
                train_loss.append(avg_epoch_loss)

                # Validate the model on the validation set
                vali_epoch_loss = self.meta_val()
                vali_loss.append(vali_epoch_loss)

                # Save the best model based on validation loss
                if vali_epoch_loss < min_loss:
                    min_loss = vali_epoch_loss
                    save_path = os.path.join(
                        self.args.save_path,
                        f'{"best___val__" + "_".join(self.args.val_datasets) + "__test__" + "_".join(self.args.test_datasets)}_model.pth'
                    )
                    print(f"Saving model at epoch {epoch}")
                    logging.info(f"Saving model at epoch {epoch}")

                    # Save model state, hyperparameters, and training info
                    torch.save({
                        'model_state_dict': copy.deepcopy(self.model.state_dict()),
                        'args': vars(self.args),  # Save hyperparameters as a dictionary
                        'epoch': epoch,
                        'min_loss': min_loss
                    }, save_path)

    def meta_val(self):
        """
        Perform meta-validation on the validation dataset.
        This function evaluates the model's ability to adapt to new tasks using a few-shot learning setup,
        similar to the training process but without updating the model's main parameters.

        Returns:
            float: Average validation loss across all meta-validation tasks.
        """
        val_loss = 0.0  # Accumulate total loss over all validation tasks

        # Set meta-batch size based on the number of validation datasets and query size
        self.args.meta_batch_size = len(self.metaval_datasets) // self.args.n_query

        # Iterate over each meta-validation task
        for task_idx in range(self.args.meta_batch_size):
            # Sample a task consisting of a support set (for adaptation) and a query set (for evaluation)
            support_set, query_set = self.metaval_datasets.sample_task(
                k_shot=self.args.k_shot,
                n_query=self.args.n_query
            )

            # Inner loop: Adapt the model on the support set using fast weights
            # Create a copy of the current model parameters (fast weights) for adaptation
            fast_weights = dict(self.model.named_parameters())
            inner_loss = 0.0

            # Perform multiple inner gradient updates (adaptation steps)
            for _ in range(self.args.num_inner_updates):
                support_loss = 0.0
                # Compute loss over all samples in the support set
                for sample in support_set:
                    x = sample['data_features'].unsqueeze(0).to(self.device)
                    d_meta_emb = sample['d_meta_embed'].unsqueeze(0).to(self.device)
                    horizon = sample['horizon'].to(self.device)
                    m_meta_emb = sample['m_meta_embed'].unsqueeze(0).to(self.device)
                    m_topo_emb = sample['m_topo'].unsqueeze(0).to(self.device)
                    m_func_emb = sample['m_func'].unsqueeze(0).to(self.device)
                    y = sample['model_rank'].unsqueeze(0).to(self.device)

                    # Forward pass using fast weights
                    prediction, mean_embed, model_embed, attn_output = self.model.functional_forward(
                        x, m_meta_emb, d_meta_emb, m_topo_emb, m_func_emb, horizon, params=fast_weights
                    )

                    # Compute ranking and MSE loss
                    rank_loss = self.criterion(prediction, y)
                    mse_loss = self.criterion_mse(prediction, y)
                    tmp_loss = self.args.rank_weight * rank_loss + self.args.mse_weight * mse_loss
                    support_loss += tmp_loss

                # Average loss over inner updates (Note: likely should be divided by len(support_set))
                inner_loss = support_loss / self.args.num_inner_updates

                # Compute gradients with respect to fast weights
                # create_graph=True allows higher-order gradients (needed for MAML-style meta-learning)
                grads = torch.autograd.grad(support_loss, fast_weights.values(), create_graph=True)

                # Update fast weights using the inner learning rate
                fast_weights = {
                    name: param - self.args.inner_lr * grad
                    for (name, param), grad in zip(fast_weights.items(), grads)
                }

            # Outer loop: Evaluate the adapted model on the query set
            query_loss = 0.0
            for sample in query_set:
                x = sample['data_features'].unsqueeze(0).to(self.device)
                d_meta_emb = sample['d_meta_embed'].unsqueeze(0).to(self.device)
                horizon = sample['horizon'].to(self.device)
                m_meta_emb = sample['m_meta_embed'].unsqueeze(0).to(self.device)
                m_topo_emb = sample['m_topo'].unsqueeze(0).to(self.device)
                m_func_emb = sample['m_func'].unsqueeze(0).to(self.device)
                y = sample['model_rank'].unsqueeze(0).to(self.device)

                # Forward pass on query set using adapted fast weights
                prediction, _, _, _ = self.model.functional_forward(
                    x, m_meta_emb, d_meta_emb, m_topo_emb, m_func_emb, horizon, params=fast_weights
                )

                # Compute loss
                rank_loss = self.criterion(prediction, y)
                mse_loss = self.criterion_mse(prediction, y)
                tmp_loss = self.args.rank_weight * rank_loss + self.args.mse_weight * mse_loss
                query_loss += tmp_loss

            # Add total query loss for this task to overall validation loss
            total_loss = query_loss
            val_loss += total_loss.item()

        # Compute average validation loss across all tasks
        avg_val_loss = val_loss / self.args.meta_batch_size
        print(f"Valid Loss: {avg_val_loss}")

        return avg_val_loss

    def meta_test(self):
        """
        Perform meta-testing on the test dataset using a pre-trained model.
        This involves loading the best model checkpoint, evaluating it on the test set,
        and calculating performance metrics for different horizons.
        """
        # Initialize the MetaModel with specified hyperparameters and move it to the device
        test_model = MetaModel(
            d_model=self.args.d_model,
            embed_dim=self.args.embed_dim,
            num_heads=self.args.num_heads,
            meta_dim=self.args.meta_dim,
            meta_outdim=self.args.meta_outdim,
            topo_dim=self.args.topo_dim,
            func_dim=self.args.func_dim,
            d_meta_dim=self.args.d_meta_dim,
            num_experts=self.args.num_experts
        ).to(self.device)

        # Load the saved checkpoint from the specified path
        checkpoint_path = os.path.join(self.args.save_path,
                                       f'{"best___val__" + "_".join(self.args.val_datasets) + "__test__" + "_".join(self.args.test_datasets)}_model.pth')
        checkpoint = torch.load(checkpoint_path)

        # Load the model weights from the checkpoint
        test_model.load_state_dict(checkpoint['model_state_dict'])
        test_model.eval()  # Set the model to evaluation mode

        # Initialize a dictionary to store results for each horizon
        horizon_results = {}
        with torch.no_grad():  # Disable gradient calculation for inference
            seen_horizons = set()  # Track all unique horizons encountered during testing

            # Iterate over batches in the test loader
            for i, batch in enumerate(self.test_loader):
                x = batch['data_features'].squeeze(0).to(self.device)
                d_meta_emb = batch['d_meta_embed'].squeeze(0).to(self.device)
                horizons = batch['horizon'].to(self.device)  # Tensor of horizons for the current batch
                m_meta_emb = batch['m_meta_embed'].squeeze(0).to(self.device)
                m_topo_emb = batch['m_topo'].to(self.device)
                m_func_emb = batch['m_func'].to(self.device)

                # Forward pass through the model
                prediction, mean_embed, model_embed, attn_output = test_model.functional_forward(
                    x, m_meta_emb, d_meta_emb, m_topo_emb, m_func_emb, horizons
                )
                y = batch['model_rank'].to(self.device)

                # Process each sample in the batch separately
                for j in range(prediction.shape[0]):
                    current_horizon = horizons[j].item()  # Horizon for this sample
                    current_pred = prediction[j].unsqueeze(0)  # Keep as 2D tensor
                    current_gt = y[j].unsqueeze(0)
                    dataset_name = batch['dataset'][j]

                    # Initialize storage for this dataset if not already present
                    if dataset_name not in horizon_results.keys():
                        horizon_results[dataset_name] = {
                            96: {'all_rank': [], 'ground_rank': [], 'losses': []},
                            192: {'all_rank': [], 'ground_rank': [], 'losses': []},
                            336: {'all_rank': [], 'ground_rank': [], 'losses': []},
                            720: {'all_rank': [], 'ground_rank': [], 'losses': []}
                        }

                    # Calculate sample-specific losses
                    rank_loss = self.criterion(current_pred, current_gt)
                    mse_loss = self.criterion_mse(current_pred, current_gt)
                    loss = self.args.rank_weight * rank_loss + self.args.mse_weight * mse_loss

                    # Store results based on horizon
                    if current_horizon in horizon_results[dataset_name]:
                        horizon_results[dataset_name][current_horizon]['all_rank'].append(current_pred)
                        horizon_results[dataset_name][current_horizon]['ground_rank'].append(current_gt)
                        horizon_results[dataset_name][current_horizon]['losses'].append(loss)
                        seen_horizons.add(current_horizon)
                    else:
                        print(f"Warning: Unexpected horizon value {current_horizon} encountered")

            # Process results for each horizon separately
            horizon_tw_results = {}

            for dataset_name in horizon_results.keys():
                if dataset_name not in horizon_tw_results.keys():
                    horizon_tw_results[dataset_name] = {}

                for horizon in seen_horizons:
                    results = horizon_results[dataset_name][horizon]
                    if results['all_rank']:  # Only process if we have data for this horizon
                        # Concatenate all predictions and ground truths for this horizon
                        all_rank = torch.cat(results['all_rank'], dim=0)
                        ground_rank = torch.cat(results['ground_rank'], dim=0)
                        avg_loss = torch.stack(results['losses']).mean()

                        # Compute Copeland's aggregation
                        agg_rank = copelands_aggregation(all_rank)

                        # Convert to dictionary format
                        agg_rank_dict = {}
                        for model in model_zoo.keys():
                            agg_rank_dict[model] = agg_rank[model_zoo[model]]

                        # Compute Kendall's W
                        tw = w_kendall_metric(agg_rank_dict, ground_rank[0].cpu().numpy())

                        # Store the result
                        horizon_tw_results[dataset_name][horizon] = tw

        return horizon_tw_results



class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=7, delta=0):
        """
        Initialize the EarlyStopping object.

        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
        """
        self.patience = patience
        self.counter = 0                   # Counter for number of epochs without improvement
        self.best_score = None             # Best score observed so far (higher is better)
        self.early_stop = False            # Flag to indicate if training should stop
        self.val_loss_min = np.Inf         # Minimum validation loss observed so far
        self.delta = delta                 # Minimum change to qualify as improvement
        self.check_point = None            # Placeholder for the best model state dict

    def __call__(self, val_loss, model, args):
        """
        Call the early stopping object with current validation loss and model.

        Args:
            val_loss (float): Current validation loss.
            model (torch.nn.Module): The model being trained.
            args (argparse.Namespace): Training arguments, including save path and dataset info.
        """
        score = -val_loss  # We want to minimize loss, so higher score (negative loss) is better

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            # Define the path to save the best model
            save_path = os.path.join(args.save_path, f'{args.test_datasets[0]}_model.pth')
            print(f"Saving model at the first epoch.")
            logging.info(f"Saving model at the first epoch.")
            # Save the model state and training arguments
            torch.save({
                'model_state_dict': model.state_dict(),
                'args': vars(args),  # Save hyperparameters as a dictionary
            }, save_path)

        elif score < self.best_score + self.delta:
            # No improvement (or improvement smaller than delta)
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            logging.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True  # Trigger early stopping
        else:
            # Improvement detected
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0  # Reset counter

    def save_checkpoint(self, val_loss, model):
        """
        Saves the model when validation loss decreases.

        Args:
            val_loss (float): Current validation loss.
            model (torch.nn.Module): The model to save.
        """
        print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        logging.info(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        # Save a deep copy of the model's state dictionary
        self.check_point = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss  # Update the minimum validation loss