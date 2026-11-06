# Module 03: Handcrafted Features

Before deep learning, computer vision meant designing measurements by hand. Many of them are
still the right tool in social science: they are transparent, cheap, and often closer to a
theoretical construct than a deep model's output. How bright is a campaign photo? How colorful is
a cereal box aimed at children? How many faces are in a protest picture, and how much text is on
a poster? This module computes those measurements and asks what each one actually captures.

## What you'll learn

- Light and color: brightness, contrast, saturation, dominant colors with k-means, and the
  colorfulness metric of Hasler and Süsstrunk
- Composition and complexity: edge density, symmetry, and compression ratio
  as a measure of visual clutter
- Faces with a classic detector: how many, how large, and where in the frame
- Visible text with OCR: how much text an image contains and what it says
- Libraries that bundle such features for social scientists, such as `athec`
- Validity: linking a feature to a construct, checking it against human judgments, and spotting
  features that measure something other than what you hoped

## Notebook

`03_handcrafted_features_exercises.ipynb` and `03_handcrafted_features_solutions.ipynb`

You compute light, color and complexity features for the Open Food Facts packaging photos and
compare products with a good and a poor Nutri-Score, extract color palettes with k-means, and see
which features measure the same thing. You validate a classic face detector on the Congress
portraits, where every image has exactly one face, before using it on crowded protest photos. You
read the text on packaging with OCR (in English and Dutch), build one feature table for module 04,
and finally rate a dozen packages yourself to see whether "colorfulness" means what you mean by
it.

## Slides

[`slides/03_handcrafted_features.pdf`](slides/03_handcrafted_features.pdf) (10 slides): why
handcrafted features still matter, light, color and colorfulness, palettes with k-means,
complexity and symmetry, the classic face detector and how to validate it, OCR, and why a precise
measure is not automatically a valid one.

The LaTeX source (`.tex`) and figures are in the same folder.

## Data and models

- The Open Food Facts, protest and Congress portrait collections from modules 01 and 02, loaded
  with the course helper (`shared/course_data.py`)
- OpenCV for edges and the classic Haar cascade face detector; scikit-learn for k-means
- Tesseract OCR with English and Dutch language packs

## Runtime

CPU. No GPU needed.

## Builds on and leads to

Builds on modules 01 and 02. Module 04 uses these features to train a classifier. Bonus module 1
of [Bags, Vectors & Transformers](https://github.com/denisejroth/bags-vectors-transformers/tree/main/bonus)
covers OCR for scanned documents in more detail.

## Further reading

- Hasler, D., & Süsstrunk, S. (2003). Measuring colorfulness in natural images. *Proceedings of
  SPIE 5007, Human Vision and Electronic Imaging VIII*, 87–95.
- Peng, Y. (2022). Athec: A Python library for computational aesthetic analysis of visual media
  in social science research. *Computational Communication Research, 4*(1).
