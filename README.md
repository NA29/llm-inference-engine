# llm-inference-engine

An LLM inference engine written from scratch in Python, PyTorch and Triton, serving
Llama-3.2-1B on a single NVIDIA A10G (24 GB).

Work in progress.

## Running on a GPU

All GPU work runs on [Modal](https://modal.com):

```bash
pip install modal && modal setup
modal run modal_app.py --cmd bench.env     # records the GPU / driver / library versions
```

Every run copies its outputs and full log back into `results/`.
