# Module 13: Video as Data

A video is a sequence of images, usually 25 or 30 per second, with sound attached. That simple
fact creates most of the practical problems of video research: an hour of footage is about
100,000 frames, most of them nearly identical. This module covers how video files work, how to
pick the frames worth analyzing, and how to turn a video into a table of shots that the methods
from Parts 1 to 3 can work on.

## What you'll learn

- Containers, codecs, frame rates and resolution, and what they mean for analysis
- FFmpeg, the tool behind almost every video workflow, and how to call it from Python
- Reading frames in Python, and keeping memory use under control
- Sampling strategies: every n-th frame, keyframes, one frame per shot, and how the choice
  changes your results
- Shot and scene detection, from simple frame differences to dedicated models
- A video as a table: one row per shot, with start, end, duration and representative frame
- Editing pace and visual change over time as measures in their own right
- Video embeddings from frame embeddings, and searching a video collection with CLIP
- Where research video comes from, and the access and copyright conditions of common sources

## Notebook

`13_video_as_data_exercises.ipynb` and `13_video_as_data_solutions.ipynb`

You download a few public domain public health films from the Internet Archive and a short
European Parliament debate clip, inspect them with FFmpeg, and extract frames with different
sampling strategies. You detect shots, build a shot table for each video, measure editing pace,
and compare the films with the debate. The last part embeds one frame per shot with CLIP and
searches all videos with a sentence, returning the matching moments with their timestamps.

## Slides

A deck on video formats, sampling and shots, with a short tour of video research in political and
health communication.

## Data and models

- Public domain films from the [Prelinger Archives](https://archive.org/details/prelinger) on the
  Internet Archive, including mid-century public health films (license checked per item)
- Debate footage from the European Parliament's
  [Multimedia Centre](https://multimedia.europarl.europa.eu/), which may be reused with
  acknowledgment of the source
- FFmpeg, PyAV, [PySceneDetect](https://www.scenedetect.com/) (BSD 3-Clause), and CLIP from
  module 09

## Runtime

Runs on a CPU; a GPU speeds up the CLIP step.

## Builds on and leads to

Builds on modules 01, 06 and 09. Modules 14 and 15 add people and sound, and module 16 combines
everything.

## Further reading

- Tarr, A., Hwang, J., & Imai, K. (2023). Automated coding of political campaign advertisement
  videos: An empirical validation study. *Political Analysis, 31*(4), 554–574.
