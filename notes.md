
# My mini Fine-Tuning notes

This section contains comprehensive notes on fine-tuning techniques, LoRA/QLoRA, and best practices.

## When to Fine-Tune

Fine-tuning is most effective for:

- **Specialized knowledge/skills** - Domain-specific expertise
- **Specific tone** - Consistent brand voice or communication style
- **Safety/guardrails** - Appropriate refusal and safety behaviors
- **Example**: GPT → ChatGPT (conversational fine-tuning)

**Decision Tree**:
- If model has the knowledge → Try prompt engineering first
- If knowledge needs to be dynamic → Try RAG (Retrieval-Augmented Generation)
- If you need specialized behavior → Fine-tune

## Supervised Fine-Tuning (SFT)

Create a specialized expert out of a generalist model.

### Data Format

Use instruction-response pairs with proper role formatting:

```python
{
  "messages": [
    {"role": "system", "content": "You are a..."},
    {"role": "user", "content": "Why is..."},
    {"role": "assistant", "content": "The..."},
    {"role": "user", "content": "Why is..."},
    {"role": "assistant", "content": "I can't answer..."}
  ]
}
```

### Key Techniques

**Compute Loss on Assistant Tokens Only**

Mask out system/user prompts during loss calculation:
- Higher accuracy, especially for multi-turn chat (~+1% in QLoRA paper)
- Less overfitting to prompts (model learns to answer, not copy questions)
- In HuggingFace: use `completion_only_collator` parameter

**Instruct vs Base Model**

- **Instruct models** - Already trained to follow instructions, require less data (head start)
- **Base models** - Need more data (e.g., 52k examples were used to train LLaMA → Alpaca)

### Dataset Requirements

**Minimum Data Size**:
- 100 high-quality training pairs (minimum)
- 1,000+ pairs (ideal)
- Diminishing returns after certain point

**Data Quality** (Priority Order):
1. **Quality over quantity** - Better to have 100 excellent examples than 1,000 mediocre ones
2. **Real over synthetic** - Consider open-source datasets on HuggingFace
3. **Structured over unstructured** - Q&A format better than raw text dumps (unless training on source code)
4. **Variety over pure quantity** - Diversity and coverage matter

### Training Parameters

**LoRA Dropout**:
- Drops random adapters during training
- Not very significant but helps if overfitting
- Start at 0.1 if needed

**Target Modules**:
- Which parts of transformer to apply adapters
- Apply to both attention and feed-forward layers for best results

**Regularization**:
- **Weight decay** (L2 penalty) - Reduces overfitting (try 0.01 to 0.1)
- **Dropout** (non-LoRA, for model activations)
- **Early stopping**

**Epochs**:
- Try lower numbers like 3 first
- Monitor for overfitting

## Reinforcement Learning (RL)

Example: RLHF (Reinforcement Learning from Human Feedback) to align models to human preferences.

### Good For

- **Alignment & safety** - SFT can do it, but RL better with reward signals (thumbs up/down)
- **Complex objectives** - Write code, solve math with correct answer AND reasoning
  - Tester can be reward function (e.g., run code and give reward +1 or 0)
- **Exploration and reasoning** - Learn to break down reasoning problems into steps
  - Supervised data may not have this info, but model can learn through RL

### Cons

- More complex and finicky than SFT
- Without clear reward function or feedback signal, RL won't work
- Training can be less stable and needs more tuning
- **Rule of thumb**: If SFT can solve it, try SFT first, then RL

### Essence

1. Model produces output
2. Reward function scores it
3. Model's parameters adjusted to get higher reward

### Data and Environment

**Environment States (Prompts)**:
- Chat: Questions the model will answer
- Math: Math problems to solve
- **No fixed answers** - Use reward function instead

**RL doesn't learn via target answers** - You must have a reward function:

- **Human preferences** - Thumbs up/down, ranking which output is better
- **Code execution** - Code runs successfully or not
- **Math verification** - "2+2=?" → Check if answer = 4

**Reward Points**:
- Can be binary: +1 or 0
- Can be signed: +1 or -1
- Can combine multiple criteria:
  - Add points for factual correctness
  - Subtract for unnecessarily long/wrong context
- Breaking rewards into parts provides more feedback than boolean of final answer

### Minimum Data Size

- Smaller datasets than SFT
- Few hundred distinct prompts can be enough
- Each prompt can be attempted repeatedly with different outputs

## LoRA & QLoRA

LoRA is applied to model weights (linear projection matrices), not to activations.

### Comparison

- **LoRA** - Faster to train, slightly better performance, needs 4x more RAM
- **QLoRA** - More memory efficient, almost same performance (slower due to quantization overhead)

**Best Practice**: LoRA should be applied to all attention and MLP layers

## QLoRA Quantization

### Introduction to Quantization

Numbers are made up of sign, range, and precision at the bit level:

```
sign:      1
range:     0011
precision: 110
Result:    10011.110 = -3.75
```

### Blockwise Quantization

Example: Quantizing 32-bit float to int8

1. int8 range: [-127, 127]
2. Chunk tensor into blocks (e.g., 2 rows)
3. Calculate quantization constant for each block:
   - `qc = 127 / absmax(block)`
   - Quantizing: `round(qc × num)`

**Why Use Blockwise?**:
- Break long tensor into smaller chunks (blocks) to quantize
- If we have very large outliers, some values may become indistinguishable
  - `[0.5, 3.0, ..., 1000.0]` → int8 → `[0, 0, ..., 127]`
- By splitting into fixed-size blocks, we reduce such errors (not fully eliminate)

### NormalFloat4 (NF4)

Neural network weights are typically in [-1, 1] and centered around 0.

**Standard quantization** = Equal bin sizes
**Normal float quantization** = Different bin sizes but equal samples per bin

- For 4-bits, divide [-1, 1] into 16 bins [0000, 0001, 0010, ..., 1111]
- QLoRA designs bins based on probability of finding a point in the bin
- Each bin has same number of points → Optimal quantization

**Quantiles**: Points taken at regular intervals from cumulative distribution function (CDF)
- E.g., median = 0.5 quantile (50% of data is below this value)
- Ensures bins are populated equally → Minimizes quantization error
- Better preserves original statistical properties

### Double Quantization

Quantization of the quantization constants themselves.

- For N blocks, there are N constants
- Blocks are groups of consecutive model parameters (weights) - artificial chunks used for quantizing

### Paged Optimizer in QLoRA

Improves memory efficiency during training.

**The Problem**:
- During training, we store model weights, gradients from backprop, and optimizer states
- Can fill up GPU fast

**The Solution**:
- Instead of keeping all optimizer states in GPU, store most in CPU RAM (larger but slower)
- When it's time to update a parameter block, optimizer states are moved back to GPU

**What is an Optimizer?**:
- Algorithm that updates NN weights during training to minimize loss function
- Gradient tells you how to adjust weights to reduce loss
- Optimizer decides how exactly to apply those gradient updates

**ADAM Optimizer**:
- Adapts learning rate per parameter
- Combines SGD, momentum, RMSprop
- Smooths noisy gradients (momentum) + adaptive scaling
- Parameters with large gradients get small steps, small gradients get larger steps

## LoRA Rank and Alpha

### Rank (r)

The inner dimension of the low-rank factorization.

- Larger rank captures more complexity
- Common values: 4, 8, 16, 32
- **Start with 16** as a good default

**Example**:
```python
# Full fine-tuning
full = (4096 × 1024) = 4,194,304 params to train

# LoRA with r = 16
A = (16, d_in) = (16, 1024)
B = (d_out, 16) = (4096, 16)
LoRA params = 16 × (1024 + 4096) = 81,920 params to train
```

### Alpha (α)

A scaling factor for the learned adapters.

- **Rule of thumb**: α = r or α = 2r
- Controls update magnitude (remember NN weights start as random numbers)
- Makes hyperparameters (learning rate, initialization) more stable when you change r
- Stabilizes training (prevents large sudden weight shifts)

## Resource Optimization Tricks

### Leverage LoRA/QLoRA
- Use parameter-efficient fine-tuning methods

### Use Smaller Models Initially
- Better to complete a 13B model than never finish a 70B

### Optimize Batch and Sequence Length
- **Reduce sequence length** - Max tokens per example (truncate/split)
- **Keep per-device batch size low** - 1-3 per batch
- **Rely on gradient accumulation** - Simulate larger batches (although slower)

### CPU Offloading
- Some frameworks allow layers to reside in CPU RAM
- Swap to GPU when it's their turn to process a batch
- Available in PyTorch

## Measuring Success

Choose metrics based on your task:

### Classification
- Accuracy, F1, Precision/Recall

### Extracting Structured Outputs (JSON)
- Exact match
- Token-level F1
- Word/Character error rate/distance

### Q&A/Summarization/Translation
- Cosine similarity
- BLEU, ROUGE, BERTScore

### Conversation
- Human evaluation on coherence, tone, adherence to instructions

### Reasoning/Math/Code/Logic
- Code execution score
- Pass@K (K different solutions for same problem, probability that at least 1 of K is correct)

### Safety/Tone Alignment
- Refusal rate
- Red-teaming
- Toxicity classifier verification

### Evaluation Layers
1. **Automated metrics** - First layer (fast, scalable)
2. **LLM as a judge** - Secondary layer (more nuanced)
3. **Human judge** - Final layer (ground truth)
