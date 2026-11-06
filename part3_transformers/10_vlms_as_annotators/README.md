# Module 10: Vision-Language Models as Annotators

A vision-language model (VLM) is a language model that can look at images. You can ask it what a
picture shows, have it fill in a coding sheet, or describe a scene in words that a text pipeline
can then analyze. Small open VLMs now run on a free Colab GPU, which makes them a realistic
annotation tool for researchers. This module treats a VLM the way you would treat a new research
assistant: give it a codebook, check its work against yours, and only trust it where it earns it.

## What you'll learn

- How VLMs are built: a vision encoder, a connector, and a language model
- Open models that fit on a free GPU, how to choose between them, and why openness about training
  data matters for research
- Prompting with images: instructions, codebooks in the prompt, few-shot examples with images
- Three kinds of output: free descriptions, labels, and structured records (JSON)
- Validation against your own codes, per category and per group
- Hallucination: when a model describes things that aren't in the image, and how to catch it
- Sensitivity checks: prompt wording, image order, image size and resolution
- Describe-then-analyze: turning images into text descriptions for a text pipeline, and what
  gets lost
- Distillation: using VLM labels to train a fast classifier on CLIP embeddings for a large
  collection

## Notebook

`10_vlms_as_annotators_exercises.ipynb` and `10_vlms_as_annotators_solutions.ipynb`

You load a small open VLM, ask it about a few images, and turn the module 04 codebook into a
prompt that returns JSON for all four variables. You code your 40 test images with it, validate
each variable against your own codes (agreement, kappa, share coded yes) and study the
disagreements. A test with leading questions shows the model describing police that aren't there,
and two sensitivity checks vary the prompt wording and the image resolution. The notebook ends with
two ways to scale up: describing images and classifying the descriptions with TF-IDF, and
distilling the VLM's codes into a linear probe on CLIP embeddings, both scored on your codes. On a
T4 the notebook takes about half an hour.

## Slides

[`slides/10_vlms_as_annotators.pdf`](slides/10_vlms_as_annotators.pdf) (11 slides): how a VLM is
built, open models that fit a free GPU, a codebook as a prompt, validation, hallucination,
sensitivity checks, describe-then-analyze versus distillation, and what to report.

The LaTeX source (`.tex`) is in the same folder.

## Data and models

- The protest collection and test set from modules 02 and 04, with all four of your codes
- `Qwen/Qwen3-VL-2B-Instruct` (Apache 2.0, no account needed). On a T4, which lacks bfloat16, it
  runs in float32. Qwen3.5 is newer but its linear-attention layers need a GPU with bfloat16;
  Molmo2 is an alternative whose training data is also open
- CLIP from module 09 for the distillation step

## Runtime

GPU needed (*Runtime → Change runtime type → T4 GPU*).

## Builds on and leads to

Builds on modules 04 and 09. Day 4 of
[Bags, Vectors & Transformers](https://github.com/denisejroth/bags-vectors-transformers) follows
the same workflow with text (prompt, validate, distill), and bonus module 15 there covers
structured outputs in more depth. Module 11 applies VLMs to posts that combine text and images,
module 16 to video, and module 17 fine-tunes one.

## Further reading

- Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual instruction tuning. *Advances in Neural
  Information Processing Systems, 36*.
- Rohrbach, A., Hendricks, L. A., Burns, K., Darrell, T., & Saenko, K. (2018). Object
  hallucination in image captioning. *Proceedings of the 2018 Conference on Empirical Methods in
  Natural Language Processing*, 4035–4045.
- Nemani, H., & Garimella, K. (2026). Large-scale multimodal content analysis and annotation with
  vision-language models. *Proceedings of the AAAI Conference on Artificial Intelligence*.
