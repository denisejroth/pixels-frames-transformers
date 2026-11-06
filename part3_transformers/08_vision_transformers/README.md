# Module 08: Vision Transformers

In 2020 researchers showed that the transformer, the architecture behind BERT and today's large
language models, works for images too if you cut each image into small patches and treat them
like words. Vision transformers (ViTs) now sit inside most state-of-the-art vision and multimodal
models, including the ones in modules 09 and 10. This module explains how they work and when they
beat the CNNs from Part 2.

## What you'll learn

- An image as a sequence of patches, and how patches become tokens
- Self-attention over patches, and what a model gains from seeing the whole image at once
- The class token, position embeddings, and what comes out of the model
- Why ViTs need more data than CNNs, and how pre-training solves that
- Self-supervised pre-training without labels: DINO-style training and masked image modeling
- Attention maps: what they show, and why they are not an explanation of a prediction
- Choosing between a CNN and a ViT for a social science task

## Notebook

`08_vision_transformers_exercises.ipynb` and `08_vision_transformers_solutions.ipynb`

You cut a protest photo into 16 × 16 patches by hand, then follow it through a pre-trained ViT:
the hidden states layer by layer, the class token, and the position embeddings, which turn out to
have learned the image's two-dimensional layout on their own. You plot attention maps and attention
rollout for a few photos and discuss what they can't show. Finally you fine-tune the ViT on the
protest labels, compare it with a linear probe on its frozen embeddings, and score both on your 40
coded images next to the methods from modules 04 to 06.

## Slides

[`slides/08_vision_transformers.pdf`](slides/08_vision_transformers.pdf) (10 slides): patches as
tokens, BERT and ViT side by side, attention over all patches and what it costs, why ViTs need
more data, self-supervised pre-training (DINOv2, masked image modeling), the limits of attention
maps, and choosing between a CNN and a ViT.

The LaTeX source (`.tex`) and the figure are in the same folder.

## Data and models

- The protest collection and test set from modules 02 and 04
- `google/vit-base-patch16-224` (Apache 2.0), pre-trained on ImageNet-21k and fine-tuned on
  ImageNet-1k
- The DINOv2 results from module 06, on your scoreboard

## Runtime

GPU needed (*Runtime → Change runtime type → T4 GPU*).

## Builds on and leads to

Builds on modules 05 and 06. Day 3 of
[Bags, Vectors & Transformers](https://github.com/denisejroth/bags-vectors-transformers) explains
attention and BERT for text; this module assumes no more than its first deck. Module 09 connects a
vision transformer to a text encoder.

## Further reading

- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ...
  Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale.
  *International Conference on Learning Representations*.
- Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., ... Bojanowski, P.
  (2024). DINOv2: Learning robust visual features without supervision. *Transactions on Machine
  Learning Research*.
