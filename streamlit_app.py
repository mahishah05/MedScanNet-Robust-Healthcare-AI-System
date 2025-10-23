"""Streamlit app for medical image classification and multimodal document question answering."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision import models

import pdfplumber
import docx
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import base64

try:
    import pytesseract  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully at runtime
    pytesseract = None

try:  # Optional imports for medical imaging formats.
    import nibabel as nib  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully at runtime
    nib = None

try:
    import pydicom  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully at runtime
    pydicom = None


@dataclass(frozen=True)
class ModelSpec:
    """Configuration needed to rebuild a model architecture for inference."""

    key: str
    name: str
    description: str
    default_weights: Path
    class_names: Tuple[str, ...]
    image_size: Tuple[int, int]
    normalization_mean: Tuple[float, float, float]
    normalization_std: Tuple[float, float, float]
    builder: Callable[[int], nn.Module]
    extra_details: str = ""

    def resolve_weights_path(self) -> Path:
        """Resolve the weights path, optionally overridden via environment variable."""
        override = os.getenv(f"{self.key.upper()}_MODEL_PATH")
        if override:
            candidate = Path(override).expanduser()
        else:
            candidate = self.default_weights
        return candidate.resolve()

@dataclass
class DocumentChunk:
    text: str
    source: str


@dataclass
class KnowledgeBase:
    chunks: List[DocumentChunk] = field(default_factory=list)
    embeddings: np.ndarray | None = None
    classification_notes: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.chunks

    def set_embeddings(self, embeddings: np.ndarray) -> None:
        self.embeddings = embeddings

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Tuple[DocumentChunk, float]]:
        if self.embeddings is None or self.embeddings.size == 0:
            return []
        similarities = cosine_similarity(query_embedding.reshape(1, -1), self.embeddings)[0]
        top_indices = similarities.argsort()[::-1][:top_k]
        return [(self.chunks[idx], float(similarities[idx])) for idx in top_indices]


HAS_TESSERACT = pytesseract is not None

def build_brain_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    return model


def build_bone_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b3(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(num_features, num_classes),
    )
    return model


MODEL_SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        key="brain",
        name="Brain Tumor Classifier",
        description="EfficientNet-B0 fine-tuned to detect glioma, meningioma, pituitary tumors, and healthy scans.",
        default_weights=Path("brain_tumor_finetuned_model.pth"),
        class_names=("glioma", "meningioma", "notumor", "pituitary"),
        image_size=(224, 224),
        normalization_mean=(0.485, 0.456, 0.406),
        normalization_std=(0.229, 0.224, 0.225),
        builder=build_brain_model,
        extra_details="The model mirrors the training notebook: EfficientNet-B0 with the final classifier replaced by a 4-class linear head.",
    ),
    ModelSpec(
        key="bone",
        name="Bone Fracture Classifier",
        description="EfficientNet-B3 trained to recognise fractures versus healthy bone radiographs.",
        default_weights=Path("bone_fracture_final_model.pth"),
        class_names=("fractured", "healthy"),
        image_size=(300, 300),
        normalization_mean=(0.485, 0.456, 0.406),
        normalization_std=(0.229, 0.224, 0.225),
        builder=build_bone_model,
        extra_details="Matches the final training configuration: EfficientNet-B3 with dropout and a 2-unit linear classifier.",
    ),
)

MODEL_REGISTRY = {spec.key: spec for spec in MODEL_SPECS}

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SUPPORTED_MEDICAL_SUFFIXES = {".dcm", ".nii", ".nii.gz"}
STREAMLIT_UPLOAD_TYPES = sorted({suffix.lstrip(".") for suffix in SUPPORTED_IMAGE_SUFFIXES | SUPPORTED_MEDICAL_SUFFIXES})

DOCUMENT_SUFFIXES = {
    "images": SUPPORTED_IMAGE_SUFFIXES | SUPPORTED_MEDICAL_SUFFIXES,
    "pdf": {".pdf"},
    "docx": {".docx"},
    "text": {".txt"},
}
DOC_UPLOAD_TYPES = sorted({suffix.lstrip(".") for values in DOCUMENT_SUFFIXES.values() for suffix in values})

GROQ_MODEL_NAME = "llama-4-11b-maverick"

SYSTEM_PROMPT = (
    "You are an AI medical documentation assistant. Analyze the provided context snippets and imaging model "
    "assessments to explain findings in clear, empathetic language. Organize answers with concise paragraphs or "
    "bullets, avoid speculation beyond the context, and clearly mark any uncertainties. Always include a short "
    "disclaimer that you are not a substitute for professional medical advice. Cite supporting sources using the "
    "format [Source: filename] matching the supplied source names."
)

SCAN_ROUTER_PROMPT = (
    "You are a radiology triage expert. Given a single medical image, identify whether it is primarily a brain scan "
    "(MRI/CT), a bone or skeletal study (X-ray/CT), or neither. Reply with one lowercase word: 'brain', 'bone', or "
    "'other'. If uncertain, reply 'other'."
)


def load_class_names(weights_path: Path, fallback: Sequence[str]) -> Tuple[str, ...]:
    """Load class names from an adjacent JSON file, if available."""
    meta_path = weights_path.with_suffix(".json")
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text())
            if isinstance(data, dict) and "class_names" in data:
                names = data["class_names"]
            else:
                names = data
            if isinstance(names, (list, tuple)) and all(isinstance(x, str) for x in names):
                return tuple(names)
            st.warning(f"Ignoring malformed metadata in {meta_path.name}; falling back to defaults.")
        except Exception as exc:  # pragma: no cover - runtime safety
            st.warning(f"Could not read {meta_path.name}: {exc}. Using default class names.")
    return tuple(fallback)


def _read_dicom(bytes_buffer: bytes) -> Image.Image:
    if pydicom is None:
        raise RuntimeError("pydicom is not installed; cannot process DICOM files.")
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        tmp.write(bytes_buffer)
        tmp_path = Path(tmp.name)
    try:
        ds = pydicom.dcmread(tmp_path)
        arr = ds.pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    finally:
        tmp_path.unlink(missing_ok=True)


def _read_nifti(bytes_buffer: bytes) -> Image.Image:
    if nib is None:
        raise RuntimeError("nibabel is not installed; cannot process NIfTI files.")
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp.write(bytes_buffer)
        tmp_path = Path(tmp.name)
    try:
        nii = nib.load(tmp_path)
        data = np.asarray(nii.get_fdata(), dtype=np.float32)
        if data.ndim == 4:  # Use first channel if multi-modal
            data = data[..., 0]
        slice_idx = data.shape[2] // 2
        arr = data[:, :, slice_idx]
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    finally:
        tmp_path.unlink(missing_ok=True)


def load_image(uploaded_file) -> Image.Image:
    """Load an uploaded image into a PIL RGB image."""
    file_bytes = uploaded_file.read()
    if not file_bytes:
        raise ValueError("Empty file uploaded.")

    return load_image_from_bytes(uploaded_file.name, file_bytes)


def load_image_from_bytes(file_name: str, data: bytes) -> Image.Image:
    suffixes = [s.lower() for s in Path(file_name).suffixes]
    primary_suffix = suffixes[-1] if suffixes else ""
    compound_suffix = "".join(suffixes[-2:]) if len(suffixes) >= 2 else primary_suffix

    if primary_suffix in SUPPORTED_IMAGE_SUFFIXES:
        return Image.open(io.BytesIO(data)).convert("RGB")
    if primary_suffix == ".dcm":
        return _read_dicom(data)
    if primary_suffix == ".nii" or compound_suffix == ".nii.gz":
        return _read_nifti(data)
    raise ValueError(f"Unsupported file type: {primary_suffix or 'unknown'}")


def build_transform(spec: ModelSpec) -> T.Compose:
    return T.Compose(
        [
            T.Resize(spec.image_size),
            T.ToTensor(),
            T.Normalize(mean=spec.normalization_mean, std=spec.normalization_std),
        ]
    )


@st.cache_resource
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


@st.cache_resource
def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Set the GROQ_API_KEY environment variable to enable chatting.")
    return Groq(api_key=api_key)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int = 220, overlap: int = 40) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def ocr_image(pil_image: Image.Image) -> str:
    if not HAS_TESSERACT:
        raise RuntimeError(
            "pytesseract is not installed. Install Tesseract OCR and the pytesseract Python package for image text extraction."
        )
    return pytesseract.image_to_string(pil_image.convert("RGB"))


def identify_scan_type(pil_image: Image.Image) -> str:
    """Use Groq vision model to route between bone, brain, or other."""

    client = None
    try:
        client = get_groq_client()
    except Exception:
        return "other"

    buffer = io.BytesIO()
    pil_image.convert("RGB").save(buffer, format="JPEG", quality=92)
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    messages = [
        {"role": "system", "content": SCAN_ROUTER_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Classify this medical image."},
                {
                    "type": "input_image",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ],
        },
    ]

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=messages,
            temperature=0,
            max_tokens=16,
        )
        content = response.choices[0].message.content.strip().lower()
    except Exception:
        return "other"

    if "bone" in content:
        return "bone"
    if "brain" in content or "tumor" in content:
        return "brain"
    return "other"


def get_specs_for_scan_type(scan_type: str | None, fallback_to_all: bool = False) -> List[ModelSpec]:
    if scan_type is None:
        return list(MODEL_SPECS)
    scan_type = scan_type.lower()
    if scan_type == "brain":
        return [MODEL_REGISTRY["brain"]]
    if scan_type == "bone":
        return [MODEL_REGISTRY["bone"]]
    return list(MODEL_SPECS) if fallback_to_all else []


def extract_text_chunks(file_name: str, data: bytes) -> List[DocumentChunk]:
    suffix = Path(file_name).suffix.lower()
    chunks: List[DocumentChunk] = []

    try:
        if suffix in DOCUMENT_SUFFIXES["pdf"]:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = clean_text(page.extract_text() or "")
                    for piece in chunk_text(text):
                        chunks.append(DocumentChunk(text=piece, source=f"{file_name} (page {page_num})"))
        elif suffix in DOCUMENT_SUFFIXES["docx"]:
            document = docx.Document(io.BytesIO(data))
            full_text = "\n".join(paragraph.text for paragraph in document.paragraphs)
            text = clean_text(full_text)
            for piece in chunk_text(text):
                chunks.append(DocumentChunk(text=piece, source=file_name))
        elif suffix in DOCUMENT_SUFFIXES["text"]:
            text = clean_text(data.decode("utf-8", errors="ignore"))
            for piece in chunk_text(text):
                chunks.append(DocumentChunk(text=piece, source=file_name))
        elif suffix in DOCUMENT_SUFFIXES["images"]:
            pil_image = load_image_from_bytes(file_name, data)
            text = clean_text(ocr_image(pil_image))
            for piece in chunk_text(text):
                chunks.append(DocumentChunk(text=piece, source=f"{file_name} (ocr)"))
    except Exception as exc:  # pragma: no cover - handled via UI feedback
        raise RuntimeError(str(exc)) from exc

    return [chunk for chunk in chunks if chunk.text]


def classify_medical_image(
    file_name: str,
    pil_image: Image.Image,
    device: torch.device,
    scan_type: str | None = None,
    fallback_to_all: bool = True,
) -> List[str]:
    summaries: List[str] = []
    target_specs = get_specs_for_scan_type(scan_type, fallback_to_all=fallback_to_all)

    if scan_type and not target_specs:
        return [
            f"Groq triage could not confidently determine whether {file_name} was a brain or bone study, so model "
            "inference was skipped."
        ]

    for spec in target_specs:
        try:
            class_names, probabilities = evaluate_spec(spec, pil_image, device)
        except Exception as exc:
            summaries.append(f"{spec.name} failed: {exc}")
            continue

        order = np.argsort(probabilities)[::-1]
        top_idx = int(order[0])
        top_prob = float(probabilities[top_idx])
        top_label = class_names[top_idx]
        top_items = ", ".join(
            f"{class_names[idx]} {probabilities[idx]:.1%}" for idx in order[: min(4, len(order))]
        )

        qualifier = "inconclusive" if top_prob < 0.5 else "predicted"
        summaries.append(
            f"{spec.name} ({'auto-routed' if scan_type else 'manual'}) analysis for {file_name}: {qualifier} "
            f"{top_label} with {top_prob:.1%} confidence. Top probabilities: {top_items}."
        )

    return summaries


def build_knowledge_base_from_files(
    files: Iterable, device: torch.device
) -> Tuple[KnowledgeBase, Dict[str, Any]]:
    kb = KnowledgeBase()
    file_summaries: List[Dict[str, Any]] = []
    errors: List[str] = []

    for uploaded_file in files:
        data = uploaded_file.getvalue()
        name = uploaded_file.name
        suffix = Path(name).suffix.lower()
        chunk_count = 0

        if data:
            try:
                chunks = extract_text_chunks(name, data)
                kb.chunks.extend(chunks)
                chunk_count += len(chunks)
            except Exception as exc:
                errors.append(f"{name}: {exc}")

            if suffix in DOCUMENT_SUFFIXES["images"]:
                try:
                    pil_image = load_image_from_bytes(name, data)
                    triage_enabled = bool(os.getenv("GROQ_API_KEY"))
                    scan_type = identify_scan_type(pil_image) if triage_enabled else None

                    if triage_enabled:
                        if scan_type == "bone":
                            routing_badge = "Bone scan detected by Groq triage."
                        elif scan_type == "brain":
                            routing_badge = "Brain scan detected by Groq triage."
                        else:
                            routing_badge = "Scan type not confidently identified; model inference skipped."
                    else:
                        routing_badge = "Groq triage disabled; ran full classifier suite."

                    if routing_badge:
                        kb.classification_notes.append(f"{name}: {routing_badge}")
                        kb.chunks.append(DocumentChunk(text=routing_badge, source=f"{name} (triage)"))
                        chunk_count += 1

                    summaries = classify_medical_image(
                        name,
                        pil_image,
                        device,
                        scan_type=scan_type,
                        fallback_to_all=not triage_enabled,
                    )
                    for summary in summaries:
                        kb.classification_notes.append(summary)
                        kb.chunks.append(DocumentChunk(text=summary, source=f"{name} (model analysis)"))
                        chunk_count += 1
                except Exception as exc:
                    errors.append(f"{name} imaging analysis failed: {exc}")

        file_summaries.append({"name": name, "chunk_count": chunk_count})

    if kb.is_empty():
        raise RuntimeError("No readable content was extracted from the uploaded documents.")

    embedding_model = get_embedding_model()
    embeddings = embedding_model.encode(
        [chunk.text for chunk in kb.chunks], convert_to_numpy=True, normalize_embeddings=True
    )
    kb.set_embeddings(embeddings)

    metadata = {"file_summaries": file_summaries, "errors": errors}
    return kb, metadata


def generate_chat_response(
    query: str,
    kb: KnowledgeBase,
    chat_history: List[Dict[str, str]],
    model_name: str = GROQ_MODEL_NAME,
) -> Tuple[str, List[Tuple[str, float]]]:
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    retrieved = kb.retrieve(query_embedding, top_k=4)

    context_segments = []
    for chunk, _ in retrieved:
        context_segments.append(f"Source: {chunk.source}\nContent: {chunk.text}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if kb.classification_notes:
        messages.append(
            {
                "role": "system",
                "content": "Imaging model assessments:\n" + "\n".join(kb.classification_notes),
            }
        )
    if context_segments:
        messages.append({"role": "system", "content": "Context passages:\n" + "\n\n".join(context_segments)})

    messages.extend({"role": item["role"], "content": item["content"]} for item in chat_history[-6:])
    messages.append({"role": "user", "content": query})

    client = get_groq_client()
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        max_tokens=800,
    )

    answer = response.choices[0].message.content.strip()
    citations = [(chunk.source, score) for chunk, score in retrieved]
    return answer, citations


@st.cache_resource(show_spinner="Loading model weights...")
def load_model(spec_key: str, weights_path: str, class_names: Tuple[str, ...], device_str: str) -> nn.Module:
    spec = MODEL_REGISTRY[spec_key]
    model = spec.builder(len(class_names))
    state_dict = torch.load(weights_path, map_location=device_str)
    model.load_state_dict(state_dict)
    model.to(torch.device(device_str))
    model.eval()
    return model


def run_inference(model: nn.Module, tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        logits = model(tensor.to(device))
        probabilities = torch.softmax(logits, dim=1)
    return probabilities.squeeze(0).cpu().numpy()


def evaluate_spec(spec: ModelSpec, pil_image: Image.Image, device: torch.device) -> Tuple[Tuple[str, ...], np.ndarray]:
    weights_path = spec.resolve_weights_path()
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights for {spec.name} not found at {weights_path}")

    class_names = load_class_names(weights_path, spec.class_names)
    model = load_model(spec.key, str(weights_path), class_names, device.type)
    transform = build_transform(spec)
    tensor = transform(pil_image).unsqueeze(0)
    probabilities = run_inference(model, tensor, device)
    return class_names, probabilities


def render_prediction(
    probabilities: np.ndarray,
    class_names: Sequence[str],
    threshold: float,
    heading: str,
) -> None:
    order = np.argsort(probabilities)[::-1]
    ordered_probs = probabilities[order]
    ordered_labels = [class_names[idx] for idx in order]

    top_probability = float(ordered_probs[0])
    top_label = ordered_labels[0].replace("_", " ").title()
    confidence_label = f"{top_probability * 100:.2f}% confidence"
    status_class = "ok" if top_probability >= threshold else "warn"

    rows_html = "".join(
        f"<tr><td>{class_names[idx].replace('_', ' ').title()}</td><td>{probabilities[idx] * 100:.2f}%</td></tr>"
        for idx in order[: min(len(order), 4)]
    )

    card_html = f"""
    <div class="prediction-card">
        <div class="prediction-card__heading">{heading}</div>
        <div class="prediction-card__result">
            <span class="prediction-card__label">{top_label}</span>
            <span class="prediction-card__confidence {status_class}">{confidence_label}</span>
        </div>
        <table class="prediction-card__table">
            <thead><tr><th>Class</th><th>Probability</th></tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)

    if top_probability < threshold:
        st.warning("The top prediction confidence is below the selected acceptance threshold.")


def main() -> None:
    st.set_page_config(page_title="Medical Imaging Copilot", layout="wide")

    st.markdown(
        """
        <style>
        :root {
            --deep-navy: #071633;
            --deep-ink: #0f2347;
            --card-bg: rgba(8, 21, 45, 0.65);
            --border-soft: rgba(140, 198, 255, 0.35);
            --accent: #38bdf8;
            --accent-strong: #22d3ee;
            --text-soft: #e4ecff;
        }
        .stApp {
            background: radial-gradient(circle at 20% 20%, rgba(56, 189, 248, 0.08), transparent 60%),
                        radial-gradient(circle at 80% 0%, rgba(34, 211, 238, 0.1), transparent 55%),
                        linear-gradient(130deg, #050d21, #081a33 55%, #0b2447);
            color: var(--text-soft);
        }
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(7, 22, 51, 0.95), rgba(7, 22, 51, 0.88));
            border-right: 1px solid rgba(120, 182, 255, 0.25);
        }
        div[data-testid="stSidebar"] * {
            color: #edf5ff !important;
        }
        .sidebar-title {
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.18rem;
            opacity: 0.7;
            margin-top: 1.5rem;
        }
        .app-hero {
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.15), rgba(59, 130, 246, 0.05));
            border: 1px solid var(--border-soft);
            border-radius: 22px;
            padding: 2.2rem 2.6rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 24px 54px rgba(6, 16, 35, 0.55);
        }
        .app-hero h1 {
            margin: 0;
            color: #f5f9ff;
            font-size: 2.4rem;
            font-weight: 700;
        }
        .app-hero p {
            margin: 1rem 0 0;
            font-size: 1.05rem;
            max-width: 760px;
            color: rgba(229, 237, 255, 0.85);
        }
        .section-card {
            background: var(--card-bg);
            border: 1px solid var(--border-soft);
            border-radius: 18px;
            padding: 1.6rem 1.9rem;
            box-shadow: 0 16px 36px rgba(4, 12, 28, 0.45);
            margin-bottom: 1.2rem;
        }
        .section-card h3 {
            margin-top: 0;
            color: #f2f7ff;
            font-weight: 600;
        }
        .scan-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.45rem 0.85rem;
            border-radius: 999px;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08rem;
            background: rgba(56, 189, 248, 0.13);
            border: 1px solid rgba(56, 189, 248, 0.35);
            margin-bottom: 0.8rem;
        }
        .scan-pill.bone {
            background: rgba(34, 197, 94, 0.15);
            border-color: rgba(34, 197, 94, 0.35);
        }
        .scan-pill.brain {
            background: rgba(168, 85, 247, 0.18);
            border-color: rgba(168, 85, 247, 0.4);
        }
        .scan-pill.other {
            background: rgba(250, 204, 21, 0.18);
            border-color: rgba(250, 204, 21, 0.35);
        }
        .prediction-card {
            background: rgba(15, 35, 71, 0.78);
            border: 1px solid rgba(148, 197, 255, 0.35);
            border-radius: 18px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 16px 32px rgba(4, 12, 28, 0.35);
        }
        .prediction-card__heading {
            font-weight: 600;
            font-size: 1.05rem;
            color: #cfe3ff;
            margin-bottom: 0.75rem;
        }
        .prediction-card__result {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 0.8rem;
        }
        .prediction-card__label {
            font-size: 1.35rem;
            font-weight: 600;
            color: #ffffff;
        }
        .prediction-card__confidence {
            font-size: 0.95rem;
            font-weight: 500;
            padding: 0.4rem 0.75rem;
            border-radius: 999px;
            border: 1px solid transparent;
        }
        .prediction-card__confidence.ok {
            background: rgba(56, 189, 248, 0.18);
            border-color: rgba(56, 189, 248, 0.4);
        }
        .prediction-card__confidence.warn {
            background: rgba(250, 204, 21, 0.18);
            border-color: rgba(250, 204, 21, 0.4);
        }
        .prediction-card__table {
            width: 100%;
            border-collapse: collapse;
        }
        .prediction-card__table th,
        .prediction-card__table td {
            text-align: left;
            padding: 0.45rem 0.2rem;
            color: #e9f2ff;
        }
        .prediction-card__table tbody tr {
            border-top: 1px solid rgba(255, 255, 255, 0.08);
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border: 1px solid transparent;
            border-radius: 999px;
            padding: 0.35rem 1.1rem;
            color: rgba(216, 231, 255, 0.7);
            font-weight: 500;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(56, 189, 248, 0.15);
            color: #e8f3ff;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: rgba(56, 189, 248, 0.22);
            color: #ffffff;
            border-color: rgba(56, 189, 248, 0.4);
        }
        .stButton button {
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.9), rgba(59, 130, 246, 0.85));
            color: white;
            border: none;
            border-radius: 999px;
            padding: 0.45rem 1.25rem;
            font-weight: 600;
            box-shadow: 0 14px 26px rgba(8, 145, 178, 0.3);
        }
        .stButton button:hover {
            background: linear-gradient(135deg, rgba(34, 211, 238, 0.95), rgba(14, 165, 233, 0.9));
        }
        .chat-citation {
            font-size: 0.8rem;
            opacity: 0.75;
        }
        .section-card ul {
            margin: 0.2rem 0 0 1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "knowledge_base" not in st.session_state:
        st.session_state["knowledge_base"] = None
        st.session_state["kb_metadata"] = {}
        st.session_state["kb_signature"] = None
        st.session_state["chat_history"] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.sidebar.markdown(f"<div class='sidebar-title'>Runtime</div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"**Device:** {device.type.upper()}")

    threshold = st.sidebar.slider(
        "Confidence alert threshold", min_value=0.1, max_value=0.99, value=0.6, step=0.01
    )

    st.sidebar.markdown("<div class='sidebar-title'>Model Library</div>", unsafe_allow_html=True)
    for spec in MODEL_SPECS:
        st.sidebar.markdown(
            f"**{spec.name}**<br/><span style='font-size:0.8rem; opacity:0.75;'>Input {spec.image_size[0]}×{spec.image_size[1]} &bull; {len(spec.class_names)} classes</span>",
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("<div class='sidebar-title'>Groq</div>", unsafe_allow_html=True)
    st.sidebar.caption(
        "Set `GROQ_API_KEY` in your environment to enable scan routing and the RAG assistant."
    )

    st.markdown(
        """
        <div class="app-hero">
            <h1>Medical Imaging Copilot</h1>
            <p>Upload diagnostic imagery and supporting documents to receive contextualised predictions, rapid summaries, and cited responses powered by EfficientNet and Groq's multimodal Llama 4 Maverick.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["🔍 Image Studio", "📚 Document Copilot"])
    image_tab, doc_tab = tabs

    with image_tab:
        st.markdown(
            """
            <div class="section-card">
                <h3>Image Analysis</h3>
                <p>Upload a radiology image and the copilot will auto-route it through a Groq triage step before running the appropriate EfficientNet model. You can adjust the confidence alert threshold from the sidebar.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_image = st.file_uploader(
            "Upload medical image",
            type=STREAMLIT_UPLOAD_TYPES,
            label_visibility="collapsed",
            key="image-uploader",
        )

        if uploaded_image is None:
            st.info("Drop an MRI, CT, or X-ray image to begin.")
        else:
            try:
                pil_image = load_image(uploaded_image)
            except Exception as exc:
                st.error(f"Failed to load image: {exc}")
            else:
                st.image(pil_image, caption="Uploaded image", use_column_width=True)

                triage_enabled = bool(os.getenv("GROQ_API_KEY"))
                if triage_enabled:
                    scan_type = identify_scan_type(pil_image)
                    pill_text = f"Scan triage: {scan_type.upper()}"
                    pill_class = scan_type
                else:
                    scan_type = None
                    pill_text = "Scan triage disabled (set GROQ_API_KEY)"
                    pill_class = "other"

                st.markdown(
                    f"<div class='scan-pill {pill_class}'>{pill_text}</div>",
                    unsafe_allow_html=True,
                )

                specs_to_run = get_specs_for_scan_type(scan_type, fallback_to_all=not triage_enabled)
                if triage_enabled and not specs_to_run:
                    st.warning("Groq triage could not determine the modality; skipping model inference.")
                else:
                    for spec in specs_to_run:
                        try:
                            class_names, probabilities = evaluate_spec(spec, pil_image, device)
                        except FileNotFoundError as missing_weights:
                            st.error(str(missing_weights))
                            continue
                        except Exception as exc:
                            st.error(f"{spec.name} failed: {exc}")
                            continue

                        render_prediction(probabilities, class_names, threshold, heading=spec.name)

                st.caption(
                    "Groq's Llama 4 Maverick provides automated modality routing prior to EfficientNet inference."
                )

    with doc_tab:
        st.markdown(
            """
            <div class="section-card">
                <h3>Document Intelligence Workspace</h3>
                <p>Load clinical reports, imaging studies, or supporting PDFs to build a local knowledge base. Groq's model powers multimodal Q&amp;A with citations, while EfficientNet summaries are injected whenever bone or brain imaging is detected.</p>
                <ul>
                    <li>Supports PDF, DOCX, TXT, PNG, JPG, TIFF, DICOM, and NIfTI.</li>
                    <li>OCR requires a local Tesseract installation.</li>
                    <li>Chat responses always include source provenance.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not HAS_TESSERACT:
            st.warning(
                "Image OCR is disabled because Tesseract is not installed. Install Tesseract and the pytesseract package for full functionality."
            )

        doc_files = st.file_uploader(
            "Add documents",
            accept_multiple_files=True,
            type=DOC_UPLOAD_TYPES,
            key="document-uploader",
        )

        clear_clicked = st.button("Clear knowledge base", key="clear-kb")
        if clear_clicked:
            st.session_state["knowledge_base"] = None
            st.session_state["kb_metadata"] = {}
            st.session_state["kb_signature"] = None
            st.session_state["chat_history"] = []
            st.info("Knowledge base cleared.")

        if doc_files:
            signature = tuple((file.name, file.size) for file in doc_files)
            if st.session_state.get("kb_signature") != signature:
                with st.spinner("Building knowledge base..."):
                    try:
                        kb, metadata = build_knowledge_base_from_files(doc_files, device)
                    except Exception as exc:
                        st.error(f"Knowledge base build failed: {exc}")
                    else:
                        st.session_state["knowledge_base"] = kb
                        st.session_state["kb_metadata"] = metadata
                        st.session_state["kb_signature"] = signature
                        st.session_state["chat_history"] = []
                        st.success(
                            f"Knowledge base ready with {len(kb.chunks)} textual segments and {len(kb.classification_notes)} imaging summaries."
                        )

        kb: KnowledgeBase | None = st.session_state.get("knowledge_base")
        metadata: Dict[str, Any] = st.session_state.get("kb_metadata", {})

        if kb:
            file_summaries = metadata.get("file_summaries", [])
            if file_summaries:
                st.markdown("#### Document coverage")
                st.dataframe(file_summaries, hide_index=True)

            error_messages = metadata.get("errors", [])
            if error_messages:
                st.warning("\n".join(error_messages))

            if kb.classification_notes:
                st.markdown("#### Imaging insights")
                for note in kb.classification_notes:
                    st.markdown(f"- {note}")

            st.markdown("#### Chat with your documents")
            for message in st.session_state.get("chat_history", []):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "assistant" and message.get("citations"):
                        st.markdown(
                            "<div class='chat-citation'>Sources</div>",
                            unsafe_allow_html=True,
                        )
                        for source, score in message["citations"]:
                            st.write(f"- {source} (similarity {score:.2f})")

            user_prompt = st.chat_input("Ask a question about the uploaded documents")
            if user_prompt:
                if not os.getenv("GROQ_API_KEY"):
                    st.error("Set the GROQ_API_KEY environment variable to enable chatting.")
                else:
                    st.session_state["chat_history"].append({"role": "user", "content": user_prompt})
                    with st.spinner("Analyzing documents..."):
                        try:
                            answer, citations = generate_chat_response(
                                user_prompt,
                                kb,
                                st.session_state["chat_history"][:-1],
                            )
                        except Exception as exc:
                            st.session_state["chat_history"].pop()
                            st.error(f"Chat failed: {exc}")
                        else:
                            st.session_state["chat_history"].append(
                                {"role": "assistant", "content": answer, "citations": citations}
                            )
        else:
            st.info("Upload one or more documents to build a knowledge base and start chatting.")


if __name__ == "__main__":
    main()
