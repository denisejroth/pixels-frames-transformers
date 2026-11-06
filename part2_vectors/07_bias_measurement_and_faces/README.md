# Module 07: Bias, Measurement and Faces

Vision models learn from the images they were trained on, and those images over-represent some
people, places and situations. The result shows up as errors that are not random: a classifier
that works well on average can fail far more often for some groups. This module teaches you to
audit a model for that, and to think hard about measurements that involve people's faces.

## What you'll learn

- Where training data comes from and who is missing from it
- Auditing a classifier across groups: error rates per group, confidence intervals, and
  differences that matter for your downstream analysis
- Labels that describe people differently by group: what Schwemmer and colleagues found about
  how commercial image recognition describes men and women
- Skin tone scales (such as the Monk Skin Tone scale) versus racial and ethnic categories, and
  what each can and can't measure
- Why inferring gender, race, age or emotion from a face is both a validity problem and an
  ethical one, and the alternatives (self-reports, metadata, coding the context instead of the
  person)
- Paired designs for audits: changing one attribute of an image while holding the rest constant
- What to report when your measurement involves people

## Notebook

`07_bias_measurement_and_faces_exercises.ipynb` and `07_bias_measurement_and_faces_solutions.ipynb`

You replicate the logic of the Gender Shades audit on a small scale. First you train a classifier
on a deliberately skewed sample of faces, the way many real training sets are skewed, then audit
it on FairFace's balanced validation set and compare error rates between groups with confidence
intervals. You then retrain on a balanced sample and break the errors down by race and gender together.
Next you probe DINOv2 embeddings of all current Congress portraits for attributes from the public
metadata (party, gender, chamber, when members entered Congress) to see what they carry about
people and about how the photos were made. After a section on skin tone scales, racial categories
and what not to infer from a face, the notebook ends with a simulation in which a smile detector
with group-specific errors invents a group difference, and you correct it with a validation sample.
On a CPU the notebook uses 3,000 faces instead of about 11,000.

## Slides

[`slides/07_bias_measurement_and_faces.pdf`](slides/07_bias_measurement_and_faces.pdf) (10
slides): Gender Shades, who is in FairFace and who labeled it, the audit design, error rates with
intervals, what embeddings encode, categories versus skin tone versus identity, how differential
error shifts conclusions, and what to report.

The LaTeX source (`.tex`) is in the same folder.

## Data and models

- [FairFace](https://huggingface.co/datasets/HuggingFaceM4/FairFace) (CC BY 4.0)
- The Congress portraits from module 01 (public domain) with party, chamber and gender from the
  congress-legislators metadata
- DINOv2 embeddings as in module 06, through the course helper (`shared/course_torch.py`)

## Runtime

A GPU is recommended (*Runtime → Change runtime type → T4 GPU*).

## Builds on and leads to

Builds on module 06. The bias measurement deck on Day 2 of
[Bags, Vectors & Transformers](https://github.com/denisejroth/bags-vectors-transformers) is the
text version of the same argument. Module 09 audits CLIP, and module 18 audits image generators.

## Further reading

- Buolamwini, J., & Gebru, T. (2018). Gender shades: Intersectional accuracy disparities in
  commercial gender classification. *Proceedings of the 1st Conference on Fairness,
  Accountability and Transparency*, PMLR 81, 77–91.
- Schwemmer, C., Knight, C., Bello-Pardo, E. D., Oklobdzija, S., Schoonvelde, M., & Lockhart, J.
  W. (2020). Diagnosing gender bias in image recognition systems. *Socius, 6*.
- Kärkkäinen, K., & Joo, J. (2021). FairFace: Face attribute dataset for balanced race, gender,
  and age for bias measurement and mitigation. *Proceedings of the IEEE/CVF Winter Conference on
  Applications of Computer Vision*, 1548–1558.
