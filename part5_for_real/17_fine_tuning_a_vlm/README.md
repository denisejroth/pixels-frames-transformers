# Module 17: Fine-Tuning a Vision-Language Model

Prompting a vision-language model works well for categories the model already understands. For a
coding scheme that is specific to your project, or a visual style the model rarely saw, it can
fall short however you word the prompt. Fine-tuning teaches the model your categories from your
own coded examples. With LoRA, that fits on a free Colab GPU for a small model. This module shows
how, and how to tell whether it was worth it.

## What you'll learn

- When fine-tuning a VLM makes sense, and when prompting, few-shot examples or a distilled
  classifier are good enough
- LoRA and QLoRA: training a small set of extra weights instead of the whole model
- Turning coded images into training conversations, with the prompt masked out of the loss
- Training a small VLM on Colab, and watching for overfitting
- A fair comparison: zero-shot, few-shot, fine-tuned VLM and a distilled CLIP classifier, all on
  the same human-coded test set
- How many coded images you need, shown with a learning curve
- Side effects: what the fine-tuned model forgets, and how its outputs drift
- Saving, documenting and sharing an adapter

## Notebook

`17_fine_tuning_a_vlm_exercises.ipynb` and `17_fine_tuning_a_vlm_solutions.ipynb`

You fine-tune a small VLM with QLoRA to code the health claims on Open Food Facts packaging,
using the codes from module 11. You compare the fine-tuned model with the
prompted model from module 10 and with a distilled CLIP classifier on the same test set, train
with different amounts of data to see where the gains level off, and check what the model now
does with prompts it was not trained on.

## Slides

A deck on fine-tuning vision-language models, with a decision guide for choosing between
prompting, distillation and fine-tuning.

## Data and models

- The coded Open Food Facts collection from module 11 (photos CC BY-SA)
- A small open VLM with an Apache 2.0 license (Qwen3.5 2B or Qwen3-VL 2B)
- Hugging Face `transformers`, `peft` and `bitsandbytes`

## Runtime

GPU needed (*Runtime → Change runtime type → T4 GPU*). Training takes a few minutes per run on a
T4.

## Builds on and leads to

Builds on modules 10 and 11. Bonus module 18 of
[Bags, Vectors & Transformers](https://github.com/denisejroth/bags-vectors-transformers/tree/main/bonus)
fine-tunes a text LLM with LoRA and covers the training details in more depth.

## Further reading

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022).
  LoRA: Low-rank adaptation of large language models. *International Conference on Learning
  Representations*.
- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning
  of quantized LLMs. *Advances in Neural Information Processing Systems, 36*.
