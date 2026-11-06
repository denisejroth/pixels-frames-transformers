# Module 02: Building an Image Corpus

Most image research starts with a collection problem, not a modeling problem. Images are easy to
find and hard to use properly: they carry copyright, they show people who never agreed to be
studied, and platforms restrict what you can download. This module walks through where images
come from, what you are allowed to do with them, and how to build a corpus that you (and a
reviewer) can trust.

## What you'll learn

- Where research images come from: archives, Wikimedia Commons and Wikidata, open data portals,
  platform research access, and scraping
- Licenses in practice: public domain, the Creative Commons family, share-alike conditions, and
  what "fair use" or quotation rights do and don't cover
- People in pictures: faces as personal data under the GDPR, consent, minors, and what changes
  when you publish results
- Platform terms and research access (for example the YouTube Data API, the TikTok Research API
  and Meta's Content Library), and why most of them give you metadata rather than media
- Downloading politely, with license and source information stored next to every file
- Finding exact and near duplicates with perceptual hashing
- A corpus manifest: one table that records where every image came from, under which license,
  and what you did to it
- Sharing a corpus you can't redistribute: identifiers, features and code instead of files

## Notebook

`02_building_an_image_corpus_exercises.ipynb` and `02_building_an_image_corpus_solutions.ipynb`

You start with four licensing and privacy scenarios, then query the Wikimedia Commons API for
photos in the categories on demonstrations and protests in the Netherlands (2015 to 2024),
collect the license, author and source of every file, and download 500-pixel thumbnails with a
pause between requests. Every file gets a row in a manifest with its source, author, license,
download date and checksum. An exercise adds King's Day photos as crowds that are not protests.
You then collect breakfast cereal packaging from Open Food Facts, find exact duplicates with
checksums and near duplicates with perceptual hashes, and end by writing a short datasheet for
your corpus. The protest and packaging collections return in later modules.

## Slides

[`slides/02_building_an_image_corpus.pdf`](slides/02_building_an_image_corpus.pdf) (10 slides):
the four questions to answer before collecting, where research images come from, licenses in
practice, a decision tree for "can I use this image in my research?", people in pictures, polite
downloading and manifests, duplicates, and datasheets.

The LaTeX source (`.tex`) and figures are in the same folder.

## Data and models

- [Wikimedia Commons](https://commons.wikimedia.org/) through its public API (license per file,
  stored in the manifest)
- [Open Food Facts](https://world.openfoodfacts.org/) product photos (CC BY-SA; the database
  itself is under the ODbL)
- `imagehash` for perceptual hashing

## Runtime

CPU. No GPU needed.

## Builds on and leads to

Builds on module 01. If you have scraped text before, bonus module 3 of
[Bags, Vectors & Transformers](https://github.com/denisejroth/bags-vectors-transformers/tree/main/bonus)
covers polite HTTP requests and APIs in more depth. Module 19 picks up the corpus again at scale.

## Further reading

- Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., &
  Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM, 64*(12), 86–92.
- The [Creative Commons license chooser](https://creativecommons.org/choose/) and the Wikimedia
  Commons guide to [reusing content](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia).
