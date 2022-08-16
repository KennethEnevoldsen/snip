import torch

from snip.models import SNPDecoder, SNPEncoder


def test_snp_encoder_decoder():
    x = torch.zeros(10, 44638)  # samples, snps
    # 44638 -> 1395 is a ~32 times reduction

    # test model construction
    enc = SNPEncoder()
    dec = SNPDecoder()

    # test forward pass
    h = enc(x)
    x_hat = dec(h, enc.forward_shapes)

    assert x_hat.shape == (10, 3, 44638)  # samples, n-categories, snps
