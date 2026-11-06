# Module 01: Images as Data

To a computer, a photo is a grid of numbers. This module makes that concrete: you open images in
Python, look at the numbers underneath, change them, and see what happens to the picture. It
also asks the question the rest of the course keeps returning to: what do social scientists want
to know from images, and which parts of that can a computer measure?

## What you'll learn

- How an image is stored: pixels, channels, bit depth and resolution
- Color spaces (RGB, grayscale, HSV) and why the choice matters for measurement
- File formats and compression, and why a JPEG is never quite the original
- Metadata such as EXIF, including what it can reveal about the person who took the photo
- Loading, resizing, cropping and displaying images with Pillow, NumPy and Matplotlib
- Working with many images at once: folders, batches and contact sheets
- What social scientists study with images, and how quantitative and interpretive visual
  analysis complement each other

## Notebook

`01_images_as_data_exercises.ipynb` and `01_images_as_data_solutions.ipynb`

You start with a single portrait, print its pixel values, split it into color channels and
convert it between color spaces. Then you edit images by changing numbers directly (brightening,
cropping, flipping, resizing), compare file formats and JPEG quality, and read and strip EXIF
metadata, including from a fake phone photo with GPS coordinates and a sideways orientation tag.
The last part downloads 200 portraits, builds a contact sheet, computes the average face of the
collection, and shows that a simple brightness measure tracks when a portrait was taken more than
anything about the person in it.

## Slides

[`slides/01_images_as_data.pdf`](slides/01_images_as_data.pdf) (11 slides): why images matter in
political, health and science communication, pixels and color spaces, the `uint8` trap,
resolution and compression, metadata, collections and averages, and the difference between an
image's content, form and meaning.

The LaTeX source (`.tex`) and figures are in the same folder.

## Data and models

- Official portraits of members of the US Congress (public domain), from the
  [unitedstates/images](https://github.com/unitedstates/images) project, linked to party and
  chamber through [congress-legislators](https://github.com/unitedstates/congress-legislators)
- For the metadata exercises, a made-up phone photo built from one of the portraits

No models in this module.

## Runtime

CPU. No GPU needed.

## Leads to

Module 02 (building your own corpus) and module 03 (measuring things in pixels).

## Further reading

- Williams, N. W., Casas, A., & Wilkerson, J. D. (2020). *Images as data for social science
  research: An introduction to convolutional neural nets for image classification*. Cambridge
  University Press.
- Joo, J., & Steinert-Threlkeld, Z. C. (2022). Image as data: Automated content analysis for
  visual presentations of political actors and events. *Computational Communication Research,
  4*(1).
- Casas, A., & Williams, N. W. (2019). Images that matter: Online protests and the mobilizing
  role of pictures. *Political Research Quarterly, 72*(2), 360–375.
