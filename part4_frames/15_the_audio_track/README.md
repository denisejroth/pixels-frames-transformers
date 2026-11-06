# Module 15: The Audio Track

Half of what a video communicates is sound: what is said, who says it, and how. A transcript
captures the words, but the same sentence spoken calmly or shouted means something different, and
music can set the tone of a whole segment. This module treats the audio track as data in its own
right and prepares it to be combined with the pictures in module 16.

## What you'll learn

- Extracting and resampling audio from video files with FFmpeg
- Transcription with Whisper, with timestamps for every segment and word
- Speaker diarization: who speaks when, and how to evaluate it
- Vocal features: pitch, loudness and speaking rate, and what they have been used to measure in
  political speech
- Speech, music and silence: segmenting the soundtrack
- Multilingual audio: the original speech and its interpretation, language identification, and
  transcripts in more than one language
- Aligning audio-based measures with shots and faces on one timeline

## Notebook

`15_the_audio_track_exercises.ipynb` and `15_the_audio_track_solutions.ipynb`

You extract the audio from the European Parliament clips and the public health films, transcribe
it with word timestamps, and diarize the debate to see who speaks when. You then compute pitch,
loudness and speaking rate per speaker turn and check whether they rise during heated exchanges,
separate narration from music in the films, and end with a speaker-turn table that module 16
joins with the shot and face tables.

## Slides

A deck on audio as data, from transcripts to the voice itself.

## Data and models

- The European Parliament clips and public domain films from module 13
- [Whisper](https://github.com/openai/whisper) (MIT) or faster-whisper for transcription
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) for diarization: the code is MIT,
  but the pre-trained pipeline asks you to accept its terms with a free Hugging Face account.
  The notebook shows how to add a token through Colab's *Secrets*, and offers an alternative from
  SpeechBrain or NVIDIA NeMo (both Apache 2.0) that needs no account.
- librosa and Parselmouth for vocal features

## Runtime

A GPU is recommended (*Runtime → Change runtime type → T4 GPU*).

## Builds on and leads to

Builds on module 13. Bonus module 2 of
[Bags, Vectors & Transformers](https://github.com/denisejroth/bags-vectors-transformers/tree/main/bonus)
covers Whisper in depth (model sizes, word error rate, hallucinations, Dutch); this module assumes
you can transcribe a file and moves on to speakers and voices. Module 16 combines the audio with
the pictures.

## Further reading

- Dietrich, B. J., Hayes, M., & O'Brien, D. Z. (2019). Pitch perfect: Vocal pitch and the
  emotional intensity of congressional speech. *American Political Science Review, 113*(4),
  941–962.
- Knox, D., & Lucas, C. (2021). A dynamic model of speech for the social sciences. *American
  Political Science Review, 115*(2), 649–666.
- Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust
  speech recognition via large-scale weak supervision. *Proceedings of the 40th International
  Conference on Machine Learning*, PMLR 202, 28492–28518.
