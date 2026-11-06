# Module 18: Generated Images

Image generators changed visual research in three ways at once. Generated images now turn up in
the collections researchers study. Researchers use generators to make stimuli for experiments.
And the generators themselves have become an object of study, because they reproduce and amplify
the patterns in their training data. This module takes each in turn.

## What you'll learn

- How text-to-image models work at a high level: diffusion, text conditioning, and why the same
  prompt gives different images
- Generated images in your data: what detectors can and can't tell you, and why detection
  accuracy drops on generators a detector has never seen
- Provenance instead of detection: content credentials (C2PA), watermarks and their limits
- Generated stimuli for experiments: control over what varies, realism and manipulation checks,
  pretesting, and disclosure to participants
- Auditing a generator: prompts as treatments, many images per prompt, measuring the results
  with CLIP, a VLM and human coders, and reporting uncertainty
- What changes ethically when the people in an image don't exist

## Notebook

`18_generated_images_exercises.ipynb` and `18_generated_images_solutions.ipynb`

You generate images with a small open text-to-image model and see how the outputs vary with the
random seed and the prompt. You then test a simple real-versus-generated classifier on a benchmark
set and watch it fail on a generator it wasn't trained on. The main exercise is a small audit:
prompts that describe people in different occupations and health situations, many images per
prompt, and measurements of who appears in them with the tools from modules 07, 09 and 10,
including a check of those tools against your own coding.

## Slides

A deck on image generation, detection and provenance, generated stimuli, and audit design.

## Data and models

- A small open text-to-image model that runs on a T4 (license checked before use; several
  popular generators come with use restrictions)
- [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) for detection exercises
  (CC BY-NC 4.0: research and teaching only)
- CLIP from module 09 and the VLM from module 10 for measurement

## Runtime

GPU needed (*Runtime → Change runtime type → T4 GPU*).

## Builds on and leads to

Builds on modules 07, 09 and 10. The audit design returns to the paired-image logic from module
07, applied to a model instead of a dataset.

## Further reading

- Bianchi, F., Kalluri, P., Durmus, E., Ladhak, F., Cheng, M., Nozza, D., Hashimoto, T., Jurafsky,
  D., Zou, J., & Caliskan, A. (2023). Easily accessible text-to-image generation amplifies
  demographic stereotypes at large scale. *Proceedings of the 2023 ACM Conference on Fairness,
  Accountability, and Transparency*, 1493–1504.
- Luccioni, A. S., Akiki, C., Mitchell, M., & Jernite, Y. (2023). Stable bias: Evaluating
  societal representations in diffusion models. *Advances in Neural Information Processing
  Systems, 36*.
