# Pixels, Frames & Transformers

### A Self-Paced Course in Computational Image and Video Analysis

A hands-on introduction to analyzing images and video in Python, written for social scientists.
The course starts with what an image looks like to a computer and ends with vision-language
models that watch and listen to video. Every module comes with a notebook you can run in your
browser, so you can work through the material at your own pace.

> **Status:** this repository is under construction. Parts 1 to 3 (modules 01 to 12) have
> notebooks and slides; Parts 4 and 5 have module guides, and their materials will follow module
> by module.

**Author:** Denise J. Roth, Strategic Communication Group, Wageningen University & Research

The course is a companion to
[Bags, Vectors & Transformers](https://github.com/denisejroth/bags-vectors-transformers), a
methods workshop in computational text analysis. You don't need to have taken it, but when a
method works the same way for text and images (validation, topic models, fine-tuning), the
modules point you there instead of repeating it.

---

## What the course covers

The course follows the same arc as its text counterpart. It starts with raw pixels, moves to
vectors that capture what an image shows, then to transformers that connect images with
language, and ends with video, where pictures, sound and speech unfold over time. The advice
stays the same throughout: start with the simplest method that answers your question, add
complexity only when it pays off, and check everything against human judgment.

| Part | Theme | What it's about |
|------|-------|-----------------|
| **1** | Pixels | What an image is to a computer, building a corpus you're allowed to use, color, light, faces and text as features, coding images by hand, and classic classifiers |
| **2** | Vectors | Convolutional networks, pre-trained models and image embeddings, similarity search and visual themes, and the measurement and bias problems that come with faces |
| **3** | Transformers | Vision transformers, CLIP and zero-shot classification with words, vision-language models as annotators, text inside images and multimodal posts, and finding objects and people |
| **4** | Frames | Video as data: shots and frames, people and motion, the audio track, and combining everything on one timeline |
| **5** | Doing it for real | Fine-tuning a vision-language model, generated images, and scaling up reproducibly |

## The modules

| # | Module | GPU? |
|---|--------|------|
| | **Part 1: Pixels** | |
| 01 | [Images as Data](part1_pixels/01_images_as_data/) | no |
| 02 | [Building an Image Corpus](part1_pixels/02_building_an_image_corpus/) | no |
| 03 | [Handcrafted Features](part1_pixels/03_handcrafted_features/) | no |
| 04 | [Coding Images and Classic Classifiers](part1_pixels/04_coding_images_and_classic_classifiers/) | no |
| | **Part 2: Vectors** | |
| 05 | [Convolutional Neural Networks for Images](part2_vectors/05_cnns_for_images/) | recommended |
| 06 | [Transfer Learning and Image Embeddings](part2_vectors/06_transfer_learning_and_embeddings/) | recommended |
| 07 | [Bias, Measurement and Faces](part2_vectors/07_bias_measurement_and_faces/) | recommended |
| | **Part 3: Transformers** | |
| 08 | [Vision Transformers](part3_transformers/08_vision_transformers/) | yes |
| 09 | [Images and Words: CLIP](part3_transformers/09_clip_images_and_words/) | recommended |
| 10 | [Vision-Language Models as Annotators](part3_transformers/10_vlms_as_annotators/) | yes |
| 11 | [Text in Images and Multimodal Posts](part3_transformers/11_text_in_images_and_multimodal_posts/) | yes |
| 12 | [Detection, Segmentation and Counting](part3_transformers/12_detection_segmentation_and_counting/) | yes |
| | **Part 4: Frames** | |
| 13 | [Video as Data](part4_frames/13_video_as_data/) | optional |
| 14 | [People in Video](part4_frames/14_people_in_video/) | recommended |
| 15 | [The Audio Track](part4_frames/15_the_audio_track/) | recommended |
| 16 | [Putting Video Together](part4_frames/16_putting_video_together/) | yes |
| | **Part 5: Doing it for real** | |
| 17 | [Fine-Tuning a Vision-Language Model](part5_for_real/17_fine_tuning_a_vlm/) | yes |
| 18 | [Generated Images](part5_for_real/18_generated_images/) | yes |
| 19 | [Scaling Up and Reproducibility](part5_for_real/19_scaling_up_and_reproducibility/) | optional |

Each module folder has a `README.md` that says what the module covers, what the notebook does,
which data and models it uses, and what to read next.

## Prerequisites

Basic Python, roughly at the level of an introductory course such as DataCamp's *Introduction to
Python* (variables, lists, loops, functions). If you need a refresher, the
[intro-to-python](https://github.com/denisejroth/intro-to-python) materials cover what you need.
No experience with images, video or machine learning is needed: Part 1 starts from scratch, and
the later parts explain the deep learning they rely on.

## How the repository is organized

```
part1_pixels/             modules 01 to 04
part2_vectors/            modules 05 to 07
part3_transformers/       modules 08 to 12
part4_frames/             modules 13 to 16
part5_for_real/           modules 17 to 19
shared/                   the slide preamble and the course helpers (data, PyTorch)
requirements.txt
LICENSE
```

Every module has its own folder with a `README.md`, a `notebooks/` folder and a `slides/` folder.
The notebooks come in pairs: an `_exercises` version to work through and a `_solutions` version
to check your answers against. The slides are short PDF decks that introduce the ideas before
you open the notebook. The decks are written in LaTeX
(Beamer) with a shared preamble in `shared/`; to rebuild one, run `latexmk -pdf` in its `slides/`
folder.

## Running the notebooks

The notebooks are written for Google Colab, so you don't need to install anything. Open a
notebook here on GitHub and click the **Open in Colab** button at the top. The first cells
install whatever that notebook needs.

- Part 1 runs on a normal (CPU) runtime.
- From Part 2 on, most modules use pre-trained models, so switch on a GPU before you start:
  *Runtime → Change runtime type → T4 GPU*. The table above says which modules need one.
- Video is heavy. The Part 4 notebooks work with short clips and sample a limited number of
  frames so they fit in Colab's free tier; each notebook explains how to scale up.

The notebooks use open models that you can download without an account or API key. The one
exception is speaker diarization in module 15, where the best open model asks you to accept its
terms with a free Hugging Face account; the notebook explains how, and offers an alternative that
needs no account.

In module 04 you code 40 images yourself. Those codes become the test set for every later
module, so each method you try is scored on the same images, and a scoreboard (`my_results.csv`)
keeps track. Because Colab forgets files when a session ends, the notebooks offer to keep both in
a folder `pixels-frames-transformers` in your Google Drive; you can also download and re-upload
them yourself.

To run the notebooks on your own computer instead, use Python 3.10 or newer and install the
packages in `requirements.txt`. Video modules also need [FFmpeg](https://ffmpeg.org/).

## Some routes through the material

The modules build on each other in order, but you don't have to take all of them. Some shorter
routes:

- **I need to classify a set of images:** 01 → 06 → 09 → 10, with 04 for validation
- **I study video:** 01 → 13 → 15 → 16, adding 14 if people's faces or bodies matter
- **I study social media posts with images and text:** 02 → 09 → 10 → 11
- **I want to explore an image collection before I measure anything:** 06 (visual themes) → 04
  (a codebook) → 10 (coding at scale)
- **I audit models for bias:** 07 → 09 → 18
- **I want to understand how the models work:** 05 → 08 → 09
- **My hand-coded data is scarce:** 04 → 10 → 17

## A note on people in images

Images and video show people far more often than text names them. A face is personal data under
the GDPR, and so is a voice. Images also carry copyright that is easy to forget when a picture is
one click away. Module 02 covers what you're allowed to collect and share, and module 07 covers
why inferring someone's gender, race, age or emotions from their face is a measurement problem as
much as an ethical one. Later modules come back to both whenever they matter.

## Data and models

The course uses openly licensed data and open models, so every notebook runs without special
access. The planned sources include public domain photos of members of the US Congress,
Wikimedia Commons, Open Food Facts product photos, the FairFace dataset, public domain films from
the Internet Archive and European Parliament debates. Each module's README lists its own data
and models with their licenses.

## License and citation

The course materials are licensed under CC BY 4.0: you can share and adapt them for any purpose,
as long as you give credit. Datasets, images, videos and pre-trained models keep their own
licenses. See `LICENSE` for details.

To credit the materials:

> Roth, D. J. (2026). *Pixels, Frames & Transformers: A self-paced course in computational image
> and video analysis* [Course materials]. Wageningen University & Research.
> https://github.com/denisejroth/pixels-frames-transformers
