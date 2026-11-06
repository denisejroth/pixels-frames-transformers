"""PyTorch helpers for Pixels, Frames & Transformers (module 05 onward).

Module 05 writes these step by step. Later modules load this file so they can focus on what is
new:

    ImageTable          a Dataset of images from file paths, with integer labels
    make_loader()       a DataLoader with sensible defaults
    train_model()       train with Adam and early stopping on validation loss
    plot_history()      training and validation curves
    predict()           predicted classes and probabilities
    embed_images()      image embeddings from a pre-trained model (DINOv2 by default), cached

Module 09 adds CLIP, module 10 a vision-language model:

    clip_images()       CLIP (or SigLIP 2) embeddings of images, cached
    clip_texts()        CLIP (or SigLIP 2) embeddings of sentences
    load_vlm()          a small open vision-language model and its processor
    ask_vlm()           ask the model about one or more images, get its answer as text
    parse_json()        read a JSON record out of a model's answer
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageOps

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_rgb(path, max_side=500):
    """Open an image upright, in RGB, no larger than max_side pixels on its longest side."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


class ImageTable(Dataset):
    """Images from a list of file paths, with integer labels."""
    def __init__(self, paths, labels, transform, max_side=256):
        self.paths, self.labels, self.transform, self.max_side = list(paths), list(labels), transform, max_side

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(load_rgb(self.paths[i], self.max_side)), int(self.labels[i])


def make_loader(dataset, shuffle=False, batch_size=32):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2,
                      pin_memory=DEVICE == "cuda")


def run_epoch(model, data_loader, loss_fn, optimizer=None):
    """One pass over the data. Trains if an optimizer is given; returns (loss, accuracy)."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in data_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def train_model(model, train_loader, val_loader, epochs=5, lr=1e-3, class_weights=None, verbose=True):
    """Train with Adam; keep the weights from the epoch with the lowest validation loss."""
    model.to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=None if class_weights is None else class_weights.to(DEVICE))
    history, best_loss, best_state = [], float("inf"), None
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "train_acc": train_acc, "val_acc": val_acc})
        if val_loss < best_loss:
            best_loss, best_state = val_loss, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"epoch {epoch:2d}  train loss {train_loss:.3f}, acc {train_acc:.2f}  |  "
                  f"validation loss {val_loss:.3f}, acc {val_acc:.2f}  ({time.time() - start:.0f}s)")
    model.load_state_dict(best_state)
    return pd.DataFrame(history)


def plot_history(history, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for ax, metric in zip(axes, ["loss", "acc"]):
        ax.plot(history["epoch"], history[f"train_{metric}"], marker="o", label="training")
        ax.plot(history["epoch"], history[f"val_{metric}"], marker="o", label="validation")
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy" if metric == "acc" else "loss")
        ax.legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def predict(model, data_loader):
    """Return (predicted classes, class probabilities) for every item in a loader."""
    model.eval()
    probs = torch.cat([model(x.to(DEVICE)).softmax(dim=1).cpu() for x, _ in data_loader])
    return probs.argmax(dim=1).numpy(), probs.numpy()


_EMBEDDERS = {}


@torch.no_grad()
def embed_images(paths, model_name="facebook/dinov2-small", batch_size=32, cache=None):
    """Embed images with a pre-trained Hugging Face vision model (the pooled output, L2-normalized).

    If `cache` is a file name, embeddings are saved there and reused on the next call with the same
    paths, so later cells and later sessions don't recompute them.
    """
    from transformers import AutoImageProcessor, AutoModel
    paths = [str(p) for p in paths]
    if cache and Path(cache).exists():
        stored = np.load(cache, allow_pickle=True)
        if list(stored["paths"]) == paths and str(stored["model"]) == model_name:
            return stored["embeddings"]
    if model_name not in _EMBEDDERS:
        _EMBEDDERS[model_name] = (AutoImageProcessor.from_pretrained(model_name),
                                  AutoModel.from_pretrained(model_name).eval().to(DEVICE))
    processor, model = _EMBEDDERS[model_name]
    chunks = []
    for start in range(0, len(paths), batch_size):
        images = [load_rgb(p, 512) for p in paths[start:start + batch_size]]
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        chunks.append(model(**inputs).pooler_output.float().cpu())
    embeddings = torch.nn.functional.normalize(torch.cat(chunks), dim=1).numpy()
    if cache:
        np.savez(cache, embeddings=embeddings, paths=np.array(paths, dtype=object), model=model_name)
    return embeddings


# ── CLIP (module 09) ──────────────────────────────────────────────────────────────────────────
CLIP_MODEL = "openai/clip-vit-base-patch32"
_CLIPS = {}


def _features(output):
    """transformers 4 returns a tensor here, transformers 5 an output object holding it."""
    return output if torch.is_tensor(output) else output.pooler_output


def load_clip(model_name=CLIP_MODEL):
    from transformers import AutoModel, AutoProcessor
    if model_name not in _CLIPS:
        _CLIPS[model_name] = (AutoProcessor.from_pretrained(model_name),
                              AutoModel.from_pretrained(model_name).eval().to(DEVICE))
    return _CLIPS[model_name]


@torch.no_grad()
def clip_images(paths, model_name=CLIP_MODEL, batch_size=32, cache=None):
    """L2-normalized image embeddings from CLIP or SigLIP 2, optionally cached in a .npz file.
    `paths` can also hold PIL images (then nothing is cached)."""
    paths = [p if isinstance(p, Image.Image) else str(p) for p in paths]
    if any(isinstance(p, Image.Image) for p in paths):
        cache = None
    if cache and Path(cache).exists():
        stored = np.load(cache, allow_pickle=True)
        if list(stored["paths"]) == paths and str(stored["model"]) == model_name:
            return stored["embeddings"]
    processor, model = load_clip(model_name)
    chunks = []
    for start in range(0, len(paths), batch_size):
        images = [p if isinstance(p, Image.Image) else load_rgb(p, 512) for p in paths[start:start + batch_size]]
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        chunks.append(_features(model.get_image_features(**inputs)).float().cpu())
    embeddings = torch.nn.functional.normalize(torch.cat(chunks), dim=1).numpy()
    if cache:
        np.savez(cache, embeddings=embeddings, paths=np.array(paths, dtype=object), model=model_name)
    return embeddings


@torch.no_grad()
def clip_texts(texts, model_name=CLIP_MODEL):
    """L2-normalized text embeddings from the same model, comparable with clip_images()."""
    processor, model = load_clip(model_name)
    # SigLIP was trained on texts padded to 64 tokens, CLIP on unpadded texts
    padding = {"padding": "max_length", "max_length": 64} if "siglip" in model_name else {"padding": True}
    inputs = processor(text=list(texts), return_tensors="pt", truncation=True, **padding).to(DEVICE)
    features = _features(model.get_text_features(**inputs)).float().cpu()
    return torch.nn.functional.normalize(features, dim=1).numpy()


# ── Vision-language models (module 10) ───────────────────────────────────────────────────────
VLM_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
_VLM = {}


def load_vlm(model_name=VLM_MODEL):
    """Load a vision-language model once. On a T4, which has no bfloat16, use float32: float16
    can overflow in these models and produce empty or garbled answers."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    if model_name not in _VLM:
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=dtype).eval().to(DEVICE)
        # greedy decoding, so the same image and prompt always give the same answer
        model.generation_config.do_sample = False
        for name in ("temperature", "top_p", "top_k"):
            setattr(model.generation_config, name, None)
        _VLM[model_name] = (processor, model)
    return _VLM[model_name]


@torch.no_grad()
def ask_vlm(images, prompt, model_name=VLM_MODEL, max_new_tokens=256, max_side=768):
    """Show the model one or more images (file paths or PIL images) and a prompt; return its answer."""
    processor, model = load_vlm(model_name)
    if not isinstance(images, (list, tuple)):
        images = [images]
    content = [{"type": "image", "image": img if isinstance(img, Image.Image) else load_rgb(img, max_side)}
               for img in images]
    content.append({"type": "text", "text": prompt})
    inputs = processor.apply_chat_template([{"role": "user", "content": content}],
                                           add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt").to(DEVICE)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    answer = output[0, inputs["input_ids"].shape[1]:]
    return processor.decode(answer, skip_special_tokens=True).strip()


def parse_json(text):
    """The first JSON object in a model's answer, as a dict; None if there is no valid one."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
