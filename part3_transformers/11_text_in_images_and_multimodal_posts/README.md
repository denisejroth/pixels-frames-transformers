# Module 11: Text in Images and Multimodal Posts

Much of what social scientists study is neither pure text nor pure image. A meme says one thing in
its caption and another in its picture. A food package makes health claims in print next to a
cartoon mascot. A social media post pairs a photo with a few words that change how it reads. This
module deals with text inside images and with posts where the meaning lives in the combination.

## What you'll learn

- Getting text out of images: classic OCR versus reading with a VLM, and how to measure
  extraction quality
- Layout: where the text sits, how large it is, and why that can matter as much as what it says
- Image plus caption: when the image changes the meaning of the text, and the other way around
- Ways to combine modalities: one model that sees both, separate models whose outputs you
  combine, or describing the image in words first
- Testing what each modality adds by removing it: text only, image only, both
- Memes as a research object, and the access conditions of common meme datasets
- Packaging and advertising as multimodal persuasion

## Notebook

`11_text_in_images_and_multimodal_posts_exercises.ipynb` and
`11_text_in_images_and_multimodal_posts_solutions.ipynb`

You read the text on Open Food Facts packaging with Tesseract and with a VLM, and score both by how
much of the product name (from the database) they find, looking out for text the VLM invents. The
VLM then codes health claims on the front of each box, compared with the claims recorded as labels
in the database. Next you make twelve memes from six protest photos, each with a supportive and a
mocking caption, and test which models notice the difference (caption only, photo only, CLIP, the
VLM), then turn the design around with one caption on different photos. The notebook ends with a
text-only versus image-only versus both comparison for predicting the packaging's Nutri-Score.

## Slides

[`slides/11_text_in_images_and_multimodal_posts.pdf`](slides/11_text_in_images_and_multimodal_posts.pdf)
(10 slides): meaning in the combination, OCR versus a VLM, invented text, two imperfect benchmarks
for health claims, a controlled meme experiment, ablations, and designs for multimodal posts.

The LaTeX source (`.tex`) is in the same folder.

## Data and models

- Open Food Facts product photos and product data (photos CC BY-SA, database ODbL)
- Protest and King's Day photos from Wikimedia Commons (free licenses) for the meme set
- Tesseract, CLIP from module 09, and `Qwen/Qwen3-VL-2B-Instruct` from module 10

## Runtime

GPU needed (*Runtime → Change runtime type → T4 GPU*).

## Builds on and leads to

Builds on modules 03, 09 and 10. Module 16 applies the same thinking to video, where speech,
on-screen text and pictures run in parallel.

## Further reading

- Kiela, D., Firooz, H., Mohan, A., Goswami, V., Singh, A., Ringshia, P., & Testuggine, D.
  (2020). The hateful memes challenge: Detecting hate speech in multimodal memes. *Advances in
  Neural Information Processing Systems, 33*.
- Nemani, H., & Garimella, K. (2026). Large-scale multimodal content analysis and annotation with
  vision-language models. *Proceedings of the AAAI Conference on Artificial Intelligence*.
