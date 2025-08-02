# MoGe-Jittor

This is a Jittor implementation of Microsoft's MoGE model.

- Original PyTorch repository: https://github.com/microsoft/MoGe/tree/main  
- Jittor framework: https://github.com/Jittor/jittor  
- Hugging Face repository (contains pretrained model weights):  
  https://huggingface.co/Tianhe122/MoGe-jittor/tree/main  

> The current Jittor version uses the **MoGE v1 Large** model weights.

## Installation

First, clone this repository:

```bash
git clone https://github.com/jidiasa/MoGe_jittor.git
cd MoGe_jittor
```

Create and activate a new conda environment:

```bash
conda create -n moge_jt python=3.10
conda activate moge_jt
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Run Example

```bash
python test.py
```
