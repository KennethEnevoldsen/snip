from argparse import Namespace

import pytorch_lightning as pl

from snip.models import create_model


def test_create_model():
    config = Namespace(
        filter_factor=1,
        layers_factor=1,
        width=64,
        dropout_p=0.01,
        fc_layer_size=None,
        architecture="snpnet",
        snp_encoding="one-hot",
        watch=False,
        learning_rate=0.01,
        optimizer="adam",
        log_slow=False,
        snp_location_feature=None,
    )

    model = create_model(config)
    isinstance(model, pl.LightningModule)
