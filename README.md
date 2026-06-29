# klaus-factory

Training data generator for KlausGPT. Talks to local LLMs (LM Studio, Ollama) 
to produce SFT conversation data with optional reasoning/think blocks.

Supports multi-host parallel generation across multiple machines and backends.

## Features
- Multi-host parallel data generation (LM Studio + Ollama)
- 200+ topic seed categories for diverse conversations
- Reasoning mode with `<think>` blocks for chain-of-thought training
- Automatic conversation cleaning and preamble stripping
- Skill classification data generation for routing models

## Usage
```
python klaus_factory.py --hosts localhost:1234:lmstudio 10.1.10.50:11434:ollama --sft-only --reasoning
```
