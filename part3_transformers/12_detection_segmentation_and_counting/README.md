# Module 12: Detection, Segmentation and Counting

Sometimes the question isn't what an image shows overall but what is in it and where: how many
people are at a rally, whether police appear in the foreground, how much of a package is taken up
by a mascot. Object detection draws boxes around things, segmentation outlines them pixel by
pixel, and newer models can find almost anything you can describe in words. This module uses them
to code parts of an image rather than the whole.

## What you'll learn

- Object detection: boxes, classes and confidence scores, and the categories a standard detector
  knows
- Open-vocabulary detection: finding objects described in free text ("a police officer", "a
  banner")
- Segmentation: outlining objects precisely with a promptable segmentation model
- Counting objects and people, and why counting crowds is hard
- Measures built from regions: size, position, overlap and share of the image
- Two-level designs: detect segments first, then classify the whole image from them
- Licenses: why some popular detection libraries come with conditions that matter for research
  code you plan to share

## Notebook

`12_detection_segmentation_and_counting_exercises.ipynb` and
`12_detection_segmentation_and_counting_solutions.ipynb`

You detect people and everyday objects in the protest collection with RT-DETR, count people per
image and check a "ten or more people" rule against your crowd codes. With OWLv2 you search for
what the standard detector doesn't know (police officers, protest signs, banners, flags) and
validate the results against your police and signs codes, including how much they depend on the
threshold and the wording. On the cereal packaging, OWLv2 finds cartoon characters and SAM 2
outlines them, so you can measure how much of the box they cover and relate it to Nutri-Score.
Finally you build a two-level classifier that predicts "protest" from the detected objects, read its
coefficients, and see what adding CLIP embeddings gains and costs.

## Slides

[`slides/12_detection_segmentation_and_counting.pdf`](slides/12_detection_segmentation_and_counting.pdf)
(10 slides): detection, segmentation and counting, COCO's categories and confidence thresholds,
why counting people is hard, open-vocabulary detection, SAM 2, measures built from regions,
two-level classification, and licenses.

The LaTeX source (`.tex`) is in the same folder.

## Data and models

- The protest collection and test set (with your crowd, police and signs codes) and the Open Food
  Facts collection
- RT-DETR (`PekingU/rtdetr_r50vd`), OWLv2 (`google/owlv2-base-patch16-ensemble`) and SAM 2.1
  (`facebook/sam2.1-hiera-tiny`), all Apache 2.0

## Runtime

GPU needed (*Runtime → Change runtime type → T4 GPU*).

## Builds on and leads to

Builds on modules 04 and 09. Module 14 follows people across video frames.

## Further reading

- Scholz, S., Weidmann, N. B., Steinert-Threlkeld, Z. C., Keremoglu, E., & Goldlücke, B. (2025).
  Improving computer vision interpretability: Transparent two-level classification for complex
  scenes. *Political Analysis, 33*(2), 107–121.
- Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., ... Girshick, R.
  (2023). Segment anything. *Proceedings of the IEEE/CVF International Conference on Computer
  Vision*, 4015–4026.
