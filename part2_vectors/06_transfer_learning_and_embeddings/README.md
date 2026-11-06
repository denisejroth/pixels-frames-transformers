# Module 06: Transfer Learning and Image Embeddings

A network trained on millions of images has already learned most of what it needs to see your
images. Transfer learning reuses that knowledge: you pass each image through a pre-trained model
and keep the vector it produces, an image embedding. Those vectors make a small labeled set go a
long way, and they let you search and cluster a collection before you have labeled anything.

## What you'll learn

- Pre-trained backbones and what they were trained on: supervised ImageNet models such as ResNet,
  and self-supervised models such as DINOv2
- Image embeddings: what the vector represents and how to extract it
- Three ways to use a pre-trained model: frozen embeddings with a linear probe, fine-tuning the
  last layers, and fine-tuning everything
- How many labeled images you need, shown with a learning curve
- Similarity search: finding images that look like a given image, and spotting near duplicates
  that perceptual hashes miss
- Clustering a collection into visual themes (embed, reduce with UMAP, cluster with HDBSCAN,
  describe), and why the result depends on your choices
- Visual frames: moving from clusters to theory-driven categories

## Notebook

`06_transfer_learning_and_embeddings_exercises.ipynb` and
`06_transfer_learning_and_embeddings_solutions.ipynb`

You first ask an ImageNet model what it sees in protest photos and packaging. Then you extract
embeddings for the protest collection with two backbones (ResNet-50 and DINOv2), train linear
probes, tune one with cross-validation on the training data only, and fine-tune the last block of
ResNet-50, all scored on your 40 coded images and added to your scoreboard next to modules 04 and
05. A learning curve shows how performance grows with the number of labels. The
second half drops the labels: you search the collection by example, find near duplicates, and
cluster the Open Food Facts photos into visual themes, inspecting each cluster and rerunning with
different settings to see how stable the themes are.

## Slides

[`slides/06_transfer_learning_and_embeddings.pdf`](slides/06_transfer_learning_and_embeddings.pdf)
(9 slides): why not to start from scratch, embeddings from a pre-trained model, three ways to use
one, choosing settings without touching the test set, learning curves, search by example and near
duplicates, and visual themes from clustering (and why a cluster is not a frame).

The LaTeX source (`.tex`) is in the same folder.

## Data and models

- The protest collection and test set (modules 02 and 04) and the Open Food Facts collection
- ResNet-50 with ImageNet weights from torchvision, and DINOv2 (`facebook/dinov2-small`, Apache
  2.0, from the Hugging Face Hub, no account needed)
- `umap-learn`, scikit-learn's `HDBSCAN`, and `imagehash`

## Runtime

A GPU is recommended (*Runtime → Change runtime type → T4 GPU*).

## Builds on and leads to

Builds on modules 04 and 05. This is the image version of Day 2 and the sentence transformer
session of [Bags, Vectors & Transformers](https://github.com/denisejroth/bags-vectors-transformers);
bonus module 11 there covers the stability of clusters and topics in more depth. Module 07 asks
what these embeddings encode about people, and module 09 adds language to the same idea.

## Further reading

- Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., ... Bojanowski, P.
  (2024). DINOv2: Learning robust visual features without supervision. *Transactions on Machine
  Learning Research*.
- Zhang, H., & Peng, Y. (2024). Image clustering: An unsupervised approach to categorize visual
  data in social science research. *Sociological Methods & Research, 53*(3), 1534–1587.
- Torres, M. (2024). A framework for the unsupervised and semi-supervised analysis of visual
  frames. *Political Analysis, 32*(2), 199–220.
