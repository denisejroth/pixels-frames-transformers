# Module 04: Coding Images and Classic Classifiers

Every automated method in this course is judged against human coding, so this module starts
there. You write a short codebook for visual content, code images yourself, measure how much two
coders agree, and only then train a classifier. The classifier uses the handcrafted features from
module 03, and its limits are the reason Part 2 exists.

## What you'll learn

- Units of analysis for images: the whole image, a region, a person, or the post an image
  belongs to
- Writing a visual codebook with clear definitions, examples and edge cases
- Coding inside a notebook, and when a dedicated tool such as Label Studio is worth setting up
- Agreement for visual codes: percent agreement, Cohen's kappa and Krippendorff's alpha, and
  what disagreements tell you about your codebook
- Supervised learning on handcrafted features: train and test splits, cross-validation, logistic
  regression and random forests
- Reading a confusion matrix and looking at the images a model gets wrong
- The semantic gap: why features computed from pixels struggle with what an image means

## Notebook

`04_coding_images_and_classic_classifiers_exercises.ipynb` and
`04_coding_images_and_classic_classifiers_solutions.ipynb`

You code 40 images (protest photos mixed with King's Day crowds) with a short codebook (protest,
crowd, police, signs) using a labeling tool inside the notebook, recode 20 of them later, and
compute agreement between the two rounds. You compare your codes with the free labels that come
from the Commons categories, then train a logistic regression and a random forest on the module
03 features with those free labels, test them against your codes, and look at the mistakes. Your
codes become the test set for Parts 2 and 3, so every later method is scored on the same images.

## Slides

[`slides/04_coding_images_and_classic_classifiers.pdf`](slides/04_coding_images_and_classic_classifiers.pdf)
(10 slides): human coding as the benchmark, units of analysis, a short visual codebook, coding
tools and habits, agreement measures, why free labels are noisy, an honestly evaluated classic
classifier, and the semantic gap.

The LaTeX source (`.tex`) and figures are in the same folder.

## Data and models

- The protest and King's Day collections from module 02, loaded with the course helper
  (`shared/course_data.py`), and the features from module 03
- scikit-learn for classifiers and agreement measures; `krippendorff` for alpha; `ipywidgets`
  for the labeling tool

## Runtime

CPU. No GPU needed.

## Builds on and leads to

Builds on modules 02 and 03. Bonus module 5 of
[Bags, Vectors & Transformers](https://github.com/denisejroth/bags-vectors-transformers/tree/main/bonus)
goes deeper into sampling, agreement and validating a classifier with confidence intervals. Module
05 replaces handcrafted features with features a network learns itself.

## Further reading

- Won, D., Steinert-Threlkeld, Z. C., & Joo, J. (2017). Protest activity detection and perceived
  violence estimation from social media images. *Proceedings of the 25th ACM International
  Conference on Multimedia*.
- Krippendorff, K. (2018). *Content analysis: An introduction to its methodology* (4th ed.). Sage.
