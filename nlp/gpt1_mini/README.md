# Improving Language Understanding by Generative Pre-Training (GPT-1)

*OpenAI Technical Report, 2018* · [Notion summary](https://road-lamp-b0c.notion.site/3fa64a44030c8153a3adec88b54de3e3) · [Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) · [Official code](https://github.com/openai/finetune-transformer-lm)

## What this implements
- Causal Transformer decoder pre-trained as a language model on unlabeled text
- Fine-tuning with a linear head on the final token and an auxiliary LM loss (lambda = 0.5)
- Comparison with the same network trained from scratch on the same 40 labels

## Simplifications
2 layers and 64 dimensions instead of 12 layers and 768; synthetic review sentences instead of BooksCorpus.

## Run
```bash
python main.py
```

## Output
```text
Stage 1: generative pre-training on 4,000 unlabeled sentences
  pre-train epoch 0  LM loss 6.501
  pre-train epoch 1  LM loss 1.485
  pre-train epoch 2  LM loss 1.282
  pre-train epoch 3  LM loss 1.221
  pre-train epoch 4  LM loss 1.189
  pre-train epoch 5  LM loss 1.156

Stage 2: fine-tuning on 40 labeled sentences, test on 1,000
  from scratch          : 0.721
  pre-trained + finetune: 0.812
```
