# Part 3: Transformers

The models that changed computer vision after 2020, and the ones that brought images and language
together. You will classify images with nothing but a list of words, use a vision-language model
as an annotator and check it against your own coding, work with posts where the text and the
picture only make sense together, and find specific objects and people inside an image.

A GPU is needed for most of this part (*Runtime → Change runtime type → T4 GPU*).

| # | Module | In short |
|---|--------|----------|
| 08 | [Vision Transformers](08_vision_transformers/) | images as patches, attention, self-supervised training, ViT versus CNN on the same data |
| 09 | [Images and Words: CLIP](09_clip_images_and_words/) | one space for images and text, zero-shot classification, prompt sensitivity, searching a collection with words, concept scales |
| 10 | [Vision-Language Models as Annotators](10_vlms_as_annotators/) | how VLMs work, prompting with images, structured outputs, validation, hallucination, distilling a VLM into a cheap classifier |
| 11 | [Text in Images and Multimodal Posts](11_text_in_images_and_multimodal_posts/) | OCR versus reading with a VLM, memes and packaging, image plus caption, what each modality adds |
| 12 | [Detection, Segmentation and Counting](12_detection_segmentation_and_counting/) | finding objects, open-vocabulary detection, segmenting regions, counting people, coding parts of an image |

Module 10 mirrors Day 4 of Bags, Vectors & Transformers: the same workflow of prompting,
validating and distilling, now with images.
