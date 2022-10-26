"""Script for calling the training of the sklearn model with different training
hyperparameters."""

import logging

from hydra import compose, initialize

from snip.train_slided_autoencoder_sklearn import main

# setup logging
logging.basicConfig(level=logging.INFO)

# create configs
activations = ["relu", "identity"]
strides = [512, 256, 128, 64, 32]
compression_factor = [2, 1.5, 1.25, 1.0]
for stride in strides:
    for cf in compression_factor:
        for act in activations:
            hidden_size = int(stride / 2)
            intermediate_hidden = stride - int((stride - hidden_size) / 2)
            overrides = [
                f"data.stride={stride}",
                f"data.width={stride}",
                f"model.hidden_layer_sizes=[{hidden_size}]",
                f"project.run_name_prefix=act{act}_strid{stride}_cf{cf}_v2",
                f"model.activation={act}",
            ]

            logging.info(f"Running with overrides: {overrides}")

            with initialize(version_base=None, config_path="../src/snip/configs"):
                cfg = compose(
                    config_name="default_config_train_slided_autoencoder_sklearn.yaml",
                    overrides=overrides,
                )
                main(cfg)

            logging.info(f"Done with stride {stride}, cf {cf}, act {act}")
