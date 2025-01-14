---
license: apache-2.0
---
DNAGPT2- The Best Beginner's Guide to Gene Sequence Large Language Models

### 1. Overview
Large language models have long transcended the NLP research domain, becoming a cornerstone for AI in science. Gene sequences in bioinformatics are most similar to natural language, making the application of large models to biological sequence studies a hot research direction in recent years. The 2024 Nobel Prize in Chemistry awarded to AlphaFold for predicting protein structures has further illuminated the future path for biological research.

However, for most biologists, large models remain unfamiliar territory. Until 2023, models like GPT were niche topics within NLP research, only gaining public attention due to the emergence of ChatGPT.

Most biology + large model research has emerged post-2023, but the significant interdisciplinary gap means these studies are typically collaborative efforts by large companies and teams. Replicating or learning from this work is challenging for many researchers, as evidenced by the issues sections of top papers on GitHub.

On one hand, large models are almost certain to shape the future of biological research; on the other, many researchers hesitate at the threshold of large models. Providing a bridge over this gap is thus an urgent need.

DNAGTP2 serves as this bridge, aiming to facilitate more biologists in overcoming the large model barrier and leveraging these powerful tools to advance their work.

### 2. Tutorial Characteristics
This tutorial is characterized by:

1. **Simplicity**: Simple code entirely built using Hugging Face’s standard libraries.
2. **Simplicity**: Basic theoretical explanations with full visual aids.
3. **Simplicity**: Classic paper cases that are easy to understand.

Despite its simplicity, the tutorial covers comprehensive content, from building tokenizers to constructing GPT, BERT models from scratch, fine-tuning LLaMA models, basic DeepSpeed multi-GPU distributed training, and applying SOTA models like LucaOne and ESM3. It combines typical biological tasks such as sequence classification, structure prediction, and regression analysis, progressively unfolding.

### Target Audience:
1. Researchers and students in the field of biology, especially bioinformatics.
2. Beginners in large model learning, applicable beyond just biology.

### 3. Tutorial Outline
#### 1 Data and Environment
1.1 Introduction to Large Model Runtime Environments  
1.2 Pre-trained and Fine-tuning Data Related to Genes  
1.3 Basic Usage of Datasets Library  

#### 2 Building DNA GPT2/Bert Large Models from Scratch
2.1 Building DNA Tokenizer  
2.2 Training DNA GPT2 Model from Scratch  
2.3 Training DNA Bert Model from Scratch  
2.4 Feature Extraction for Biological Sequences Using Gene Large Models  
2.5 Building Large Models Based on Multimodal Data  

#### 3 Biological Sequence Tasks Using Gene Large Models
3.1 Sequence Classification Task  
3.2 Structure Prediction Task  
3.3 Multi-sequence Interaction Analysis  
3.4 Function Prediction Task  
3.5 Regression Tasks  

#### 4 Entering the ChatGPT Era: Gene Instruction Building and Fine-tuning
4.1 Expanding LLaMA Vocabulary Based on Gene Data  
4.2 Introduction to DeepSpeed Distributed Training  
4.3 Continuous Pre-training of LLaMA Model Based on Gene Data  
4.4 Classification Task Using LLaMA-gene Large Model  
4.5 Instruction Fine-tuning Based on LLaMA-gene Large Model  

#### 5 Overview of SOTA Large Model Applications in Biology
5.1 Application of DNABERT2  
5.2 Usage of LucaOne  
5.3 Usage of ESM3  
5.4 Application of MedGPT  
5.5 Application of LLaMA-gene