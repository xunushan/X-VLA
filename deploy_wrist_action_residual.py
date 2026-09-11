"""Inference server entry point for R0/R1 checkpoints; regular deploy.py is untouched."""

import deploy
from models.wrist_action_residual import WristActionResidualXVLA


if __name__ == "__main__":
    deploy.XVLA = WristActionResidualXVLA
    deploy.main()
