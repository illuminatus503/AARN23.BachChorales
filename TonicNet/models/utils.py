import torch
import pytorch_lightning as pl


def print_model_summary(model: pl.LightningModule):
    """
    Prints a summary of the given PyTorch Lightning model including hyperparameters,
    total parameters, trainable parameters, and size in MB.

    Args:
    model (pl.LightningModule): The PyTorch Lightning model to summarize.
    """

    # Initialize variables to store total parameters
    total_params = 0
    total_trainable_params = 0
    params_per_layer = {}

    # Iterate through each layer
    for name, parameter in model.named_parameters():
        # Count the parameters
        num_params = parameter.numel()
        total_params += num_params

        if parameter.requires_grad:
            total_trainable_params += num_params

        # Add to the dictionary
        layer_name = name.split(".")[0]  # Extract the layer name
        params_per_layer.setdefault(layer_name, 0)
        params_per_layer[layer_name] += num_params

    # Print the model's hyperparameters
    print(" ** NET HYPERPARAMETERS ** ")
    for hparam in model.hparams:
        print(f" -- {hparam} : {type(model.hparams[hparam])} {model.hparams[hparam]}")
    print()

    # Print the model's parameters summary
    print(
        f"  Total Parameters: {total_params} ({total_params * 4 / (1024 ** 2):.2f} MB)"
    )
    print(
        f"  -- Total Trainable Parameters: {total_trainable_params} ({total_trainable_params * 4 / (1024 ** 2):.2f} MB)"
    )
    print()

    # Print parameters per layer
    print(" ** PARAMETERS PER LAYER ** ")
    for layer, num_params in params_per_layer.items():
        print(
            f" -- {layer}: {num_params} parameters ({num_params * 4 / (1024**2):.2f} MB)"
        )
