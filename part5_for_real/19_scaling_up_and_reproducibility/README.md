# Module 19: Scaling Up and Reproducibility

The notebooks in this course work with hundreds of images and a few minutes of video. Real
projects work with hundreds of thousands of images or hundreds of hours of footage, and media
files are large, slow to decode and often can't be shared. This module covers what changes at
that scale, and how to set up a project that someone else can repeat.

## What you'll learn

- Storing a large image or video collection: folders, Parquet, WebDataset shards, and streaming
  datasets from the Hugging Face Hub
- Fast inference: batching, data loaders, mixed precision, and decoding video efficiently
- Computing embeddings once and caching them, so later analyses don't touch the media again
- Chunked and resumable jobs for long runs
- Pinning everything: package versions, model revisions, random seeds and data snapshots
- From Colab to a cluster: turning a notebook into a script, and running it as a job array
- Sharing what you can't redistribute: identifiers, hashes, features, codes and the code to
  rebuild the collection
- Documenting data and models with datasheets and model cards
- Compute budgets and the energy cost of large vision models

## Notebook

`19_scaling_up_and_reproducibility_exercises.ipynb` and
`19_scaling_up_and_reproducibility_solutions.ipynb`

You take the protest pipeline from Parts 1 to 3 and rebuild it for scale: the corpus stored as
shards with its manifest, embeddings computed in batches and cached to disk, a resumable job that
survives a disconnected Colab session, and pinned model revisions. You turn the notebook into a
script with a configuration file, write a job script for a SLURM cluster, and finish with a
replication package that contains everything except the images themselves.

## Slides

A deck on scale and reproducibility for media research, with a checklist for a replication
package.

## Data and models

- The protest corpus and manifest from module 02 and the models from modules 06 and 09
- `datasets`, `webdataset` and PyTorch data loaders

## Runtime

Runs on a CPU; a GPU makes the embedding step much faster.

## Builds on and leads to

Builds on modules 02, 06 and 09. Bonus module 13 of
[Bags, Vectors & Transformers](https://github.com/denisejroth/bags-vectors-transformers/tree/main/bonus)
covers the same ground for text (project structure, git, Zenodo, SLURM job arrays); this module
focuses on what is different for images and video.

## Further reading

- Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., &
  Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM, 64*(12), 86–92.
- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E.,
  Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. *Proceedings of the Conference
  on Fairness, Accountability, and Transparency*, 220–229.
