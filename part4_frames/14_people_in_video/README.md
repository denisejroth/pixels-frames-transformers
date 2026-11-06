# Module 14: People in Video

Most video that social scientists study is about people: politicians debating, experts
explaining, creators talking to a camera. Video adds what a photo can't show: who is on screen and
for how long, how people move, and how they react to each other. This module measures those
things and pays close attention to the ones that are easy to compute and hard to interpret.

## What you'll learn

- Detecting faces in every sampled frame, and following the same person across frames
- Screen time: who is visible, for how long, and in which shots
- Pose estimation: body landmarks, gestures and posture
- Motion as a measure: frame differences and optical flow, and how motion has been used to study
  political behavior
- Facial expression analysis, and why the link between facial movements and emotions is much
  weaker than many tools suggest
- Recognizing specific people: when it is defensible (public officials in their public role) and
  when it is not, and what the GDPR and the EU AI Act say about biometric data
- Aggregating person-level measures to shots, speakers and whole videos

## Notebook

`14_people_in_video_exercises.ipynb` and `14_people_in_video_solutions.ipynb`

Working with European Parliament debate clips, you detect and track faces, compute screen time
per person, and compare the camera's attention to the speaker with its attention to the audience.
You extract body pose and gesture rate for each speaker and measure motion in the chamber during
applause and interruptions. The last section runs an off-the-shelf expression model on the same
clips and asks what its "emotion" scores can and can't support.

## Slides

A deck on measuring people in video, from screen time to motion, and the validity of expression
analysis.

## Data and models

- European Parliament debate footage (reuse with acknowledgment of the source) and the public
  domain films from module 13
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) (Apache 2.0) for face detection and pose
- OpenCV for motion and optical flow
- An open facial expression toolkit for the final section

## Runtime

A GPU is recommended (*Runtime → Change runtime type → T4 GPU*).

## Builds on and leads to

Builds on modules 07, 12 and 13. Module 15 adds the audio track, and module 16 lines up who is on
screen with who is speaking.

## Further reading

- Dietrich, B. J. (2021). Using motion detection to measure social polarization in the U.S. House
  of Representatives. *Political Analysis, 29*(2), 250–259.
- Boussalis, C., Coan, T. G., Holman, M. R., & Müller, S. (2021). Gender, candidate emotional
  expression, and voter reactions during televised debates. *American Political Science Review,
  115*(4), 1242–1257.
- Barrett, L. F., Adolphs, R., Marsella, S., Martinez, A. M., & Pollak, S. D. (2019). Emotional
  expressions reconsidered: Challenges to inferring emotion from human facial movements.
  *Psychological Science in the Public Interest, 20*(1), 1–68.
