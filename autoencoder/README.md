---
datasets:
- monology/pile-uncopyrighted
language:
- en
library_name: transformers
license: mit
metrics:
- BrierLM
tags:
- large language models
- language modeling
pipeline_tag: text-generation
---
# Continuous Autoregressive Language Models

[![Paper](https://img.shields.io/badge/Paper_📃-green)](https://arxiv.org/abs/2510.27688)
[![GitHub](https://img.shields.io/badge/GitHub_🧑‍💻-blue)](https://github.com/shaochenze/calm)
[![HuggingFace](https://img.shields.io/badge/HuggingFace_🤗-orange)](https://huggingface.co/collections/cccczshao/calm)
[![Blog](https://img.shields.io/badge/Blog_✍️-yellowgreen)](https://shaochenze.github.io/blog/2025/CALM/)


## Model Description

Modern Large Language Models (LLMs) are constrained by a fundamental bottleneck: they generate text one token at a time. **CALM (Continuous Autoregressive Language Models)** confronts this challenge by introducing a paradigm shift in language modeling. Instead of predicting one discrete token at a time, CALM learns to predict a single continuous vector that represents an entire chunk of K tokens.

This is achieved through a two-stage process:

1. **A high-fidelity autoencoder** learns to compress K tokens into a single vector and reconstruct them with near-perfect accuracy.
2. **A continuous-domain language model** then performs autoregressive prediction in this vector space.

### Key Features

* 🚀 **Ultra-Efficient by Design:** Dramatically improves training and inference efficiency by reducing the number of autoregressive steps by a factor of K.
* 💡 **A New Scaling Axis:** Introduces a new scaling dimension for LLMs—semantic bandwidth (K). Instead of just scaling parameters and data, you can now scale the amount of information processed in a single step.
* 🛠️ **A Comprehensive Likelihood-Free Toolkit:** Operating in a continuous domain requires new tools. This repository provides the full suite of algorithms that make CALM possible:
  
  * **A Robust Autoencoder** to learn high-fidelity continuous representations of token chunks.
  * **Energy-Based Training**, a principled and likelihood-free method for generative modeling.
  * **BrierLM**, a new metric for calibrated, likelihood-free evaluation of language models.
  * **Temperature Sampling** for controlled, high-quality text generation using only a black-box sampler.

## How to use

See our [GitHub README](https://github.com/shaochenze/calm), where we provide scripts for training and evaluation.

## Contact

If you have any questions, feel free to submit an issue or contact `chenzeshao@tencent.com`.