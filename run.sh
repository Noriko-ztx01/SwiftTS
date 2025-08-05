#!/bin/bash

# ========================
# Single-run configuration script
# ========================

# Record script start time
start_time=$(date +%s)

# Set parameters (run only this one configuration)
inner_lr=0.001
meta_lr=0.005
mse_weight=0.7
train_epochs=80
seed=1
setting="full"
horizon_setting="all"
k_shot=16
n_query=16

# Main log directory and save path
master_log_dir="./log_meta"
param_dir="inner${inner_lr}_meta${meta_lr}_mse${mse_weight}_epoch${train_epochs}_seed${seed}_${setting}_${horizon_setting}_k${k_shot}_q${n_query}"
save_path="${master_log_dir}/${param_dir}"

# Create save directory
mkdir -p "$save_path"

# Print configuration info
echo "=== Running single configuration ==="
echo "Parameters:"
echo "  inner_lr        = $inner_lr"
echo "  meta_lr         = $meta_lr"
echo "  mse_weight      = $mse_weight"
echo "  train_epochs    = $train_epochs"
echo "  seed            = $seed"
echo "  setting         = $setting"
echo "  horizon_setting = $horizon_setting"
echo "  k_shot          = $k_shot"
echo "  n_query         = $n_query"
echo "Save Path: $save_path"

# Record start time for this run
combination_start=$(date +%s)

# Execute training command
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

# Check execution status
if [ $? -eq 0 ]; then
    combination_end=$(date +%s)
    combination_duration=$((combination_end - combination_start))
    echo "Success! Runtime: $combination_duration seconds."
    echo "Logs and models saved to $save_path"
else
    combination_end=$(date +%s)
    combination_duration=$((combination_end - combination_start))
    echo "Failed after $combination_duration seconds!"
    echo "Check log file: ${save_path}/training.log"
fi

# Calculate total runtime
end_time=$(date +%s)
total_duration=$((end_time - start_time))

# Function to format time (HH:MM:SS)
format_time() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(( (seconds % 3600) / 60 ))
    local secs=$((seconds % 60))
    printf "%02d:%02d:%02d" $hours $minutes $secs
}

# Output final summary
echo "============================================"
echo "Training complete!"
echo "Total runtime: $(format_time $total_duration)"
echo "Results saved in: $save_path"
echo "============================================"