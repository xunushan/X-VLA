import math

import torch

from train_three_camera_preclip import _grad_norm


def test_decoder_preclip_then_global_clip_preserves_total_bound():
    decoder = torch.nn.Parameter(torch.zeros(2))
    other = torch.nn.Parameter(torch.zeros(2))
    decoder.grad = torch.tensor([12.0, 9.0])  # norm 15
    other.grad = torch.tensor([0.3, 0.4])     # norm 0.5

    raw_decoder = torch.nn.utils.clip_grad_norm_([decoder], 1.0)
    after_decoder = _grad_norm([decoder, other])
    raw_final = torch.nn.utils.clip_grad_norm_([decoder, other], 1.0)

    assert math.isclose(float(raw_decoder), 15.0, rel_tol=1e-6)
    assert math.isclose(after_decoder, math.sqrt(1.0 + 0.25), rel_tol=1e-5)
    assert math.isclose(float(raw_final), after_decoder, rel_tol=1e-5)
    assert _grad_norm([decoder, other]) <= 1.00001
    # Other gradients retain almost all of their raw norm (rather than 1/15).
    assert _grad_norm([other]) > 0.44

