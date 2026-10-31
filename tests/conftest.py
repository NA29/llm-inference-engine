import os

import pytest
import torch

MODEL_ID = os.environ.get("MODEL_ID", "unsloth/Llama-3.2-1B")

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.fixture(scope="session")
def model_dir():
    from engine.model import download_model
    return download_model(MODEL_ID)
