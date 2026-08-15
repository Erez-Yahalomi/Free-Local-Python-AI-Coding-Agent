# 🖥️ Local Coding Agent

#  A free, fully local AI coding agent platform generate Python code from natural-language instructions directly on your own computer.   
Powered by Qwen2.5-Coder-7B. 

## ✨ Highlights

- 🆓 **Free to use** — no costs, no LLM/API tokens or cloud inference charges

- 🔒 **Local and Private** — your prompts and generated code are made and stayed on your machine

- 🐍Specialized for Python code generation

- 🌐 Can also be used in Google Colab with a compatible  Tesla T4 GPU

## 🏗️ How It Works

Type/prompt in the interface window, your program description. The coding agent will generate Python code, with code explanation. Then you can run the your program’s generated code separately.


## 💻 Requirements

### Hardware

- A CUDA-enabled GPU, such as: RTX 4060 Ti (16GB),  RTX 3090 / 4090 (24GB)

- CUDA-compatible GPU

- Recommended: **12 GB+ VRAM **for 4-bit quantization.

- System RAM: **16 GB+ recommended**

- Storage: **20 GB+ recommended**

### Software

- Windows 10/11 or Linux

- Python 3.12

- PyTorch (Latest Stable), library torch

- CUDA-enabled PyTorch

- `Transformers `(Hugging Face): Specifically optimized to handle Qwen2.5-Coder-7B-Instruct

- `accelerate`

- `bitsandbytes `Essential for the 4-bit (NF4) quantization logic 

## 📦 Run

```
python 

AI coding agent.py


On first run the the agent loads the Qwen coding model locally.

---------------------------------------------------------------------
  
 A private, free AI coding assistant running on your own GPU. 

 Software Created by Erez-Yahalomi ארז-יהלומי


 
