# Weights & Biases Integration

This document explains how to use the Weights & Biases (wandb) integration for experiment tracking, hyperparameter optimization, and visualization.

## Setup

1. Install the necessary packages:
```bash
pip install -r requirements.txt
```

2. Create a free wandb account at [wandb.ai](https://wandb.ai/) if you don't already have one.

3. Login to wandb from the command line:
```bash
wandb login
```

## Single Training Run with wandb

To run a training session with wandb logging:

```bash
python train_with_wandb.py --config config.yaml
```

This will:
- Initialize a new wandb run
- Train the model using the settings from `config.yaml`
- Log metrics, predictions, and model artifacts to wandb
- Show you a link to view the results in the wandb dashboard

## Hyperparameter Optimization with wandb Sweeps

To run a hyperparameter sweep:

```bash
python train_with_wandb.py --config sweep_config.yaml --sweep --count 10
```

This will:
- Create a new wandb sweep using the parameters defined in `sweep_config.yaml`
- Start a sweep agent that will run 10 training jobs with different hyperparameters
- Optimize according to the metric defined in the sweep configuration (by default `eval/mAP`)
- Show you a link to view the sweep results in the wandb dashboard

## Available wandb Utilities

The `wandb_utils.py` module provides the following functionality:

1. **Initializing wandb**:
   ```python
   from wandb_utils import init_wandb_run

   run = init_wandb_run(
       config=config,
       project_name="bibnet",
       run_name="my-experiment"
   )
   ```

2. **Creating and starting sweeps**:
   ```python
   from wandb_utils import create_sweep_config, start_sweep

   sweep_config = create_sweep_config(
       method="bayes",
       metric={"name": "eval/mAP", "goal": "maximize"},
       parameters={
           "batch_size": {"values": [4, 8, 16]},
           "lr": {"min": 0.0001, "max": 0.01, "distribution": "log_uniform"}
       }
   )

   sweep_id = start_sweep(
       sweep_config=sweep_config,
       train_function=my_train_function,
       count=5
   )
   ```

3. **Logging metrics during training**:
   ```python
   from wandb_utils import log_metrics

   # Log training metrics
   log_metrics(
       metrics={"loss": 0.123, "obj_loss": 0.045},
       prefix="train"
   )

   # Log evaluation metrics
   log_metrics(
       metrics={"mAP": 0.86, "precision": 0.92},
       prefix="eval"
   )
   ```

4. **Saving model artifacts**:
   ```python
   from wandb_utils import save_model_artifact

   save_model_artifact(
       model=my_model,
       run_id=run.id,
       artifact_name="best-model",
       metadata={"epoch": 10, "mAP": 0.85}
   )
   ```

## Customizing the Sweep Configuration

You can customize the sweep configuration by editing `sweep_config.yaml`. The file includes:

- **Method**: The search algorithm ('grid', 'random', or 'bayes')
- **Metric**: The metric to optimize and whether to maximize or minimize it
- **Parameters**: The hyperparameters to sweep over and their ranges
