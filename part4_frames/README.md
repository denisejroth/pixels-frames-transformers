# Part 4: Frames

Video as data. A video is a stack of images with sound and, often, speech, so this part reuses
almost everything from Parts 1 to 3 and adds time. You will cut videos into shots, follow people
across frames, analyze the audio track, and line everything up on one timeline that a
vision-language model or your own coding can work from.

Video is heavy, so the notebooks use short clips and sample a limited number of frames to stay
within Colab's free tier. Each notebook explains how to scale up. A GPU is recommended for
modules 14 to 16.

| # | Module | In short |
|---|--------|----------|
| 13 | [Video as Data](13_video_as_data/) | frames, frame rates and codecs, FFmpeg, sampling frames, shot detection, a video as a table of shots |
| 14 | [People in Video](14_people_in_video/) | detecting and tracking faces, screen time, pose and gesture, motion as a measure, the trouble with facial expressions |
| 15 | [The Audio Track](15_the_audio_track/) | transcripts with timestamps, who speaks when, pitch, loudness and speaking rate, music, speech and silence |
| 16 | [Putting Video Together](16_putting_video_together/) | one timeline for shots, speech, on-screen text and faces; video-language models; coding segments and whole videos; validation |

If you need transcription basics first, bonus module 2 of
[Bags, Vectors & Transformers](https://github.com/denisejroth/bags-vectors-transformers/tree/main/bonus)
covers Whisper in English and Dutch.
