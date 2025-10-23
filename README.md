# QLoRA Fine-Tuning: Qwen "Do Not Answer" Model

This repository contains a fine-tuned Qwen model trained with QLoRA (Quantized Low-Rank Adaptation) to handle "do not answer" scenarios - teaching the model when to appropriately refuse to answer certain types of questions.


This repo serve as a practice and is for the sole purpose of education. In reality, many models exist for such a use-case without the need for additional fine-tuning. One can instead opt for comprehensive guardrails at different points in the inference pipeline. One is also advice to use an instruct tuned model and a larger model for fine-tuning. 


## 📁 Repository Contents

- `SFT_training.ipynb` - Supervised fine-tuning training notebook
- `Eval.ipynb` - Evaluation on hold out set
- `Compare.ipynb` - Model comparison and evaluation notebook
- `qwen3-do-not-ans-final.zip` - Trained LoRA adapters and model configuration
- `notes.md` - Mini notes I compiled while reading up on the topic


## 📊 Training Details

- **Method**: Supervised Fine-Tuning (SFT) with QLoRA
- **Base Model**: Qwen3-0.6B
- **Quantization**: 4-bit NF4 with double quantization
- **LoRA Configuration**:
  - Rank (r): 32
  - Alpha: 64
  - Target modules: All attention and MLP layers
- **Training Focus**: Teaching appropriate refusal behaviors for unsafe questions
- **Adapters**: Available in `qwen3-do-not-ans-final/`


## Observations from Eval:
Despite being an educational practice, some positive results were observed! 
- SFT responses are almost 2× closer semantically to your ideal refusals.
- Fine-tuned model shows a much higher baseline and ceiling — meaning even its worst replies are closer to ideal than the base model’s best ones.
- Strong gain in contextual similarity — SFT model’s language and tone align with safe templates, whereas the base model diverged significantly.
- The fine-tuned model now consistently produces semantically meaningful and safe text, instead of unrelated or unsafe completions.

-----


references: 
- https://github.com/artidoro/qlora
- https://github.com/Libr-AI/do-not-answer
- https://github.com/QwenLM/Qwen3