# Module 09: Images and Words: CLIP

CLIP was trained on hundreds of millions of images paired with their captions, and learned to put
an image and a sentence that describes it close together in one shared space. That makes it
possible to classify images without any labeled examples, just by writing down what each category
looks like in words, and to search an image collection with a sentence. It also brings the
problems of language models into vision: the wording of your prompt changes your results.

## What you'll learn

- Contrastive training: how CLIP learns from image and caption pairs
- One space for images and text, and what similarity between an image and a sentence means
- Zero-shot classification: turning categories into prompts, and scoring images against them
- Prompt sensitivity and prompt ensembles, and how to report them
- Searching a collection with words, and with a mix of words and images
- Concept scales: placing images between two poles defined by text ("calm" versus "chaotic"),
  much like the poles in the embedding playground of Bags, Vectors & Transformers
- Linear probes on CLIP embeddings as a strong, cheap baseline
- What CLIP learned from the web about people, and how to audit it

## Notebook

`09_clip_images_and_words_exercises.ipynb` and `09_clip_images_and_words_solutions.ipynb`

You put images and sentences into one space, then classify your 40 coded images zero-shot. Five
prompt wordings are compared on the training images (never on the test set) before the chosen one
is scored, and a prompt ensemble and a linear probe on the same embeddings follow, with SigLIP 2 as
an alternative model. You search the Open Food Facts packaging with sentences and with an image
plus words, build a "healthy-looking" scale from two sentences and check it against the products'
Nutri-Score, and end with an audit of how CLIP labels the occupations of members of Congress,
women and men, where the right answer is known.

## Slides

[`slides/09_clip_images_and_words.pdf`](slides/09_clip_images_and_words.pdf) (10 slides):
contrastive learning, one space for images and sentences, zero-shot classification, prompt
sensitivity, zero-shot versus labels, search, scales defined in words, and auditing what CLIP
learned about people.

The LaTeX source (`.tex`) is in the same folder.

## Data and models

- The protest test set, the Open Food Facts collection (with Nutri-Score) and the Congress
  portraits
- `openai/clip-vit-base-patch32` (MIT) and SigLIP 2 (`google/siglip2-base-patch16-224`,
  Apache 2.0), through the course helper (`shared/course_torch.py`)

## Runtime

A GPU is recommended (*Runtime → Change runtime type → T4 GPU*). The base CLIP model also runs
on a CPU for small collections.

## Builds on and leads to

Builds on modules 06 to 08. Zero-shot classification of text on Day 3 of
[Bags, Vectors & Transformers](https://github.com/denisejroth/bags-vectors-transformers) uses the
same logic. Module 10 moves from scoring images against words to asking a model questions about
them.

## Further reading

- Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... Sutskever, I.
  (2021). Learning transferable visual models from natural language supervision. *Proceedings of
  the 38th International Conference on Machine Learning*, PMLR 139, 8748–8763.
