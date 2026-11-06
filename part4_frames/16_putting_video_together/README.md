# Module 16: Putting Video Together

With shots, faces, transcripts and voices measured separately, the last step is to combine them.
There are two routes. The pipeline route lines up everything on one timeline and codes segments
from that table, which is transparent and easy to validate. The end-to-end route gives the video
to a model that watches and listens at once, which is powerful and much harder to check. This
module does both on the same videos and compares them against human coding.

## What you'll learn

- One timeline: joining shot, face, speaker-turn and on-screen text tables by time
- Choosing the unit of analysis: frame, shot, speaker turn, segment or whole video
- Coding segments from the joined table, and aggregating segment codes to a video
- Video-language models: how they sample frames, how much video they can take in at once, and
  what that means for long recordings
- Asking a video-language model to code a segment with a codebook, with structured output
- Validating video codes: sampling segments for human coding, agreement over time, and errors
  that cluster in particular kinds of scenes
- When the pipeline beats the end-to-end model, and the other way around

## Notebook

`16_putting_video_together_exercises.ipynb` and `16_putting_video_together_solutions.ipynb`

You join the shot, face and speaker-turn tables from modules 13 to 15 into one segment table for
the public health films and code persuasion techniques per segment (expert on screen, a
demonstration, a direct address to the viewer, a fear appeal) with rules built on that table. You
then code the same segments with a small open video-language model, compare both approaches with a
set of human codes, and look at where each one fails.

## Slides

A deck on multimodal video analysis, with a comparison of pipeline and end-to-end designs.

## Data and models

- The public domain films and European Parliament clips from modules 13 to 15
- A small open model that accepts video, such as Qwen3.5 (2B or 4B) or Qwen3-VL (2B or 4B), both
  Apache 2.0; SmolVLM2 as a lightweight alternative

## Runtime

GPU needed (*Runtime → Change runtime type → T4 GPU*). The notebook limits the number of frames
per segment so the model fits in Colab's free tier.

## Builds on and leads to

Builds on modules 10 and 13 to 15. Module 17 shows how to fine-tune a vision-language model when
prompting isn't enough.

## Further reading

- Tarr, A., Hwang, J., & Imai, K. (2023). Automated coding of political campaign advertisement
  videos: An empirical validation study. *Political Analysis, 31*(4), 554–574.
- Nemani, H., & Garimella, K. (2026). Large-scale multimodal content analysis and annotation with
  vision-language models. *Proceedings of the AAAI Conference on Artificial Intelligence*.
