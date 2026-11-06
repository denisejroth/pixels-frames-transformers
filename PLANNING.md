# Planning notes

Working notes for building the course. Not meant for learners; remove or move before the repo
goes public.

*Last updated: November 6, 2026 (Part 3 added)*

## Build status

| Module | Notebooks | Slides | Tested |
|--------|-----------|--------|--------|
| 01 Images as Data | done (8 exercises) | done (11 slides) | both versions run end to end on real data (CPU, about 2 minutes) |
| 02 Building an Image Corpus | done (7 exercises) | done (10 slides) | logic tested against an offline stand-in for the Commons and Open Food Facts APIs; **needs one real run in Colab** |
| 03 Handcrafted Features | done (7 exercises) | done (10 slides) | face detection checked on real portraits; downloads tested offline only; **needs one real run in Colab** |
| 04 Coding Images and Classic Classifiers | done (6 exercises) | done (10 slides) | logic tested offline with a stand-in coder, including the scoreboard section; **needs one real run in Colab, with you coding** |
| 05 CNNs for Images | done (6 exercises) | done (10 slides) | runs end to end offline on CPU (19 minutes) with a fake CIFAR-10 and stand-in data; **needs one real run in Colab (GPU)** |
| 06 Transfer Learning and Embeddings | done (6 exercises) | done (9 slides) | runs end to end offline on CPU (13 minutes) with random-weight stand-ins for ResNet-50 and DINOv2; **needs one real run in Colab (GPU)** |
| 07 Bias, Measurement and Faces | done (5 exercises) | done (10 slides) | runs end to end offline on CPU (13 minutes) with a fake FairFace and random-weight DINOv2; **needs one real run in Colab (GPU)** |
| 08 Vision Transformers | done (6 exercises) | done (10 slides) | runs end to end offline on CPU (3 minutes) with a tiny random-weight ViT; **needs one real run in Colab (GPU)** |
| 09 Images and Words: CLIP | done (6 exercises) | done (10 slides) | runs end to end offline on CPU with a tiny random-weight CLIP; **needs one real run in Colab** |
| 10 VLMs as Annotators | done (6 exercises) | done (11 slides) | runs end to end offline with a fake VLM that returns plausible answers; **needs one real run in Colab (T4), and the timings in the notebook are estimates** |
| 11 Text in Images and Multimodal Posts | done (5 exercises) | done (10 slides) | runs end to end offline (real Tesseract on stand-in packaging, fake VLM); **needs one real run in Colab** |
| 12 Detection, Segmentation and Counting | done (5 exercises) | done (10 slides) | runs end to end offline with fake RT-DETR, OWLv2 and SAM 2 that mimic their APIs; **needs one real run in Colab** |
| 13 to 19 | not started | not started | |

The build workspace could reach GitHub and PyPI but not Wikimedia or Open Food Facts, which is
why modules 02 to 04 have only been tested against a stand-in that mimics the real API responses.
Things to watch for on the first real run: the number of files the Commons categories return,
whether any collected images are unsuitable for teaching (the code skips subcategories with
words like "naked" or "WNBR", but nothing else filters content), and the Open Food Facts rate
limit messages.

**Course data helper.** Modules 03 to 07 load their images through `shared/course_data.py`
(and modules 06 and 07 also `shared/course_torch.py`), which they download from the repository's
`main` branch. It repeats the collection code from module 02 in compact form, plus the feature
function from module 03. Until the repository is on GitHub, those notebooks stop at the download
with a 404 error, unless the helpers are copied next to them.

**Freezing the corpora.** Commons categories and Open Food Facts change over time, so different
learners will get slightly different images. `course_data.py` already checks for frozen
manifests at `shared/data/protests_manifest.csv`, `packaging_manifest.csv` and
`koningsdag_manifest.csv` and downloads exactly those files when they exist. To freeze: run
module 02 once, look through the images, drop anything unsuitable, and commit the three
manifests.

**Part 2 testing.** The build workspace has no GPU and can't reach Hugging Face, PyTorch's model
server or the CIFAR-10 host. Modules 05 to 07 were run end to end on CPU against stand-ins with the
same interfaces (a fake CIFAR-10 and FairFace, ResNet-50 and DINOv2 with random weights), which
checks that every cell runs but says nothing about the numbers. Solution comments that describe
results are hedged ("usually", "often") for that reason; check them against a real run.

**Part 3 testing.** Same approach. ViT and CLIP stand-ins are real transformers classes with tiny
random configurations; the VLM, RT-DETR, OWLv2 and SAM 2 stand-ins are small fakes with the same
call signatures, written after checking the installed transformers 5.22 source and the model docs.
Things to watch on the first real run: whether `facebook/sam2.1-hiera-tiny` loads with
`Sam2Model` (the transformers docs use the `sam2.1-hiera-large` checkpoint); Qwen3-VL's speed in
float32 on a T4 (the notebook says about half an hour for module 10); OWLv2 box coordinates (recent
transformers versions correct for OWLv2's square padding, older ones don't, so the notebooks require
transformers 4.57 or newer); and whether the solution comments about typical results hold.

**Part 3 decisions.** Qwen3-VL-2B-Instruct is the default VLM, run in float32 on a T4: Qwen3.5's
linear-attention layers depend on a Triton kernel that fails to compile for bfloat16 weights on the
T4, and float16 is known to overflow in Qwen VL models. The CLIP and VLM helpers live in
`shared/course_torch.py`. The COCO caption sample planned for module 09 was dropped; search is shown
on the packaging collection instead, and the "healthy-looking" scale is checked against Nutri-Score.
Memes in module 11 are made in the notebook from Commons protest photos and never shared.
`course_data` asks Open Food Facts for `labels_tags`, which module 11 uses as the benchmark for
health claims.

**Shared test set and scoreboard.** From module 04 on, `course_data.protest_train_test()` gives
every module the same split: the learner's 40 coded images as the test set and every other image
(with its category label) for training. `keep_my_files()` copies the codes and the scoreboard
(`my_results.csv`) to Google Drive, and later modules restore them. If the codes are missing, the
test set falls back to category labels with a loud warning. `shared/course_torch.py` holds the
module 05 training loop and a cached DINOv2 embedding function for later modules.

**Second coder (was open question 3).** Module 04 now measures stability instead: learners
recode 20 of their 40 images later in the notebook and compare. It also explains how to do the
same with a colleague for inter-coder reliability. No author-provided codes are needed.

**Slides.** One Beamer deck per module, in the BVT bonus format (about ten slides, ending with an
"In the Notebook" slide). Sources, figures and the compiled PDF sit in each module's `slides/`
folder; `shared/preamble.tex` is the BVT preamble with the course name and a `\coursemodule`
label, and needs the `fontawesome5` package. Figures with photos use only the public domain
Congress portraits (credited on each slide); the OCR example is a made-up sign. The numbers on
the slides (JPEG sizes, hash distances, the 94% face detection rate, brightness by period) were
computed on the real portraits, so they match what the module 01 and 03 notebooks produce.

## Decisions so far

- **Format:** self-paced, 19 modules in five parts, no day structure.
- **Audio:** part of the video material (module 15), not a separate track.
- **Conventions carried over from Bags, Vectors & Transformers:** Colab first, exercise and
  solution notebook pairs, short Beamer decks in WUR style (`shared/preamble.tex` is the BVT
  preamble), open and ungated models wherever possible, CC BY 4.0 for the materials, and one test
  set that every method is scored on so results compare across modules.
- **Framework:** PyTorch and Hugging Face throughout. BVT's bonus modules used Keras on the
  PyTorch backend; dropping Keras keeps the stack smaller here.
- **Overlap with BVT:** link rather than repeat. Bonus 2 (Whisper) feeds module 15, bonus 5
  (validation) feeds module 04, bonus 6 and 9 (neural nets, CNNs for images) feed module 05,
  bonus 11 (BERTopic) feeds module 06, bonus 13 (reproducibility) feeds module 19, bonus 15
  (structured outputs) feeds module 10, bonus 18 (LoRA) feeds module 17.

## Running data threads

BVT worked because the bill titles ran through most of the course. The proposal here is three
threads, each openly licensed:

| Thread | Used in | Why |
|--------|---------|-----|
| **Protest and demonstration photos** from Wikimedia Commons | 02, 03, 04, 05, 06, 08, 09, 10, 12, 19 | A classic task in political communication; the hand-coded test set from module 04 scores every method in Parts 1 to 3 |
| **Open Food Facts packaging** | 02, 03, 06, 09, 11, 12, 17 | Health communication; text in images; product metadata (categories, claims, Nutri-Score) gives labels without hand coding |
| **Public domain health films and EP debates** | 13, 14, 15, 16 | Openly reusable video with narration, people and multilingual speech |

Faces get their own sources (Congress portraits, FairFace) so that the main threads don't depend
on inferring things about people.

## Data: license status

| Source | License or terms | Checked? | Notes |
|--------|------------------|----------|-------|
| [unitedstates/images](https://github.com/unitedstates/images) (Congress portraits) | Public domain (US government works) | Yes | Repo updated May 2026; metadata from congress-legislators (CC0) |
| [FairFace](https://huggingface.co/datasets/HuggingFaceM4/FairFace) | CC BY 4.0 | Yes | Balanced by age, gender and race group; needed for module 07 |
| [Open Food Facts](https://world.openfoodfacts.org/) | Photos CC BY-SA; database ODbL | Yes | Share-alike: any derived image set we host must be CC BY-SA too. Some packaging shows brand logos and characters that carry their own rights |
| Wikimedia Commons | Per file (public domain, CC BY, CC BY-SA, ...) | Per file | Store license and author per image in the manifest; filter out non-free files |
| European Parliament audiovisual material | Reuse authorized with acknowledgment of the source; partial reproductions should link to the full item | Yes, via the EP legal notice as quoted in the Europarl-ASR license | Confirm on the Multimedia Centre before building; check whether interpretation audio channels can be downloaded with the video |
| [Prelinger Archives](https://archive.org/details/prelinger) | Mostly public domain | Per item | Check the license field of each film we use |
| [COCO](https://cocodataset.org/) | Annotations CC BY 4.0; images under various Flickr licenses | Partly | Dropped from module 09; RT-DETR in module 12 is trained on it |
| [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) | CC BY-NC 4.0 | Yes | Fine for teaching; very large (about 2.45M rows), so stream a small slice |
| Konstanz Protest Image Dataset | Images on request only | Yes | Not usable in open notebooks; cite Scholz et al. (2025) in module 12 instead |
| Hateful Memes | Meta's dataset license, accepted on download | Not in detail | Mention in module 11, don't build on it |

## Models: shortlist

All should run on Colab's free T4 (16 GB, no bf16), and none should need an account unless noted.

| Use | Candidates | License | Checked? | Notes |
|-----|-----------|---------|----------|-------|
| VLM, images and video | Qwen3.5 2B and 4B (March 2026) | Apache 2.0 | Yes | Natively multimodal, accepts video; fails on a T4 (Triton can't compile its bf16 Gated DeltaNet kernel for sm75), so not the default |
| VLM, images and video | Qwen3-VL 2B and 4B Instruct (October 2025) | Apache 2.0 | Yes | Not gated; needs transformers 4.57+; **used in modules 10 and 11**, float32 on a T4 |
| VLM, open training data | Molmo2 4B (January 2026) | Apache 2.0 | Yes | Ai2 releases the training data too, a good teaching point |
| Small VLM with video | SmolVLM2 (256M to 2.2B) | Likely Apache 2.0 | To verify | Fallback when memory is tight |
| Image backbone, self-supervised | DINOv2 small and base | Apache 2.0 | From memory | DINOv3 is gated behind Meta's form and its own license, so skip it |
| Image backbone, supervised | ResNet-50 (torchvision), ViT-B/16 | BSD / Apache 2.0 | From memory | `google/vit-base-patch16-224` in module 08 |
| Image and text | CLIP ViT-B/32; SigLIP 2 | MIT; Apache 2.0 | From memory | In transformers 5, `get_image_features` returns an output object; the helper handles both |
| Detection | RT-DETR; OWLv2 or Grounding DINO | Apache 2.0 | Yes (RT-DETR and OWLv2 model cards) | `PekingU/rtdetr_r50vd` and `google/owlv2-base-patch16-ensemble` in module 12; avoid Ultralytics YOLO as the default: AGPL-3.0 |
| Segmentation | SAM 2 | Apache 2.0 | From memory | `facebook/sam2.1-hiera-tiny` with `Sam2Model` in module 12; from memory: SAM 3 adds text prompts but comes with Meta's own license and gated access, so check before switching |
| Faces and pose | MediaPipe | Apache 2.0 | From memory | |
| Shots | PySceneDetect; TransNetV2 | BSD 3-Clause; MIT | From memory | |
| Transcription | Whisper, faster-whisper | MIT | From memory | |
| Diarization | pyannote 3.1 / community pipeline | Code MIT, models gated | Yes | Free Hugging Face account and token needed; offer SpeechBrain or NeMo (Apache 2.0, ungated) as the no-account route |
| Text-to-image | To decide | Varies | No | Many popular generators have use restrictions or don't fit a T4 at full precision |

The small-VLM landscape moves every few months. Checked again for module 10 in November 2026; recheck
before writing modules 16 and 17.

## Open questions

1. **Title.** Working title is *Pixels, Frames & Transformers*. Alternatives: *Pixels, Patches &
   Prompts*, or keep the BVT rhythm with *Pixels, Vectors & Frames*.
2. **Hosting the course data.** Curated subsets (the protest set with its manifest and test-set
   codes, the packaging set) would be easiest to load from the Hugging Face Hub under your
   account, with license metadata per file. The Open Food Facts subset would have to be CC BY-SA.
3. **Dutch material.** BVT had Dutch OCR and transcription. Worth adding Dutch sources here (for
   example Tweede Kamer footage, if its terms allow reuse)?

## Suggested build order

Part 1 first, since it creates the corpus and test set everything else uses. Then module 06 and
09 (the backbone of most later work), then 10. Video (Part 4) can be built in parallel once
module 13's data choices are settled.
