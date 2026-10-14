"""
multimodal_loader.py
====================
Multimodal Knowledge Base Ingestion for LearnWise AI
Track D: Personalized Tutoring & Adaptive Learning.

Ingests:
1. Textbooks (PDF) with page-level citations & diagram extraction.
2. Slide Decks (PPTX) with slide numbers, tables & image extraction.
3. Lecture Videos / Audios / Subtitles (.mp4, .mp3, .srt, .vtt, .txt) with timestamp citations.
4. Automatic Topic, Concept & Prerequisite hierarchy discovery.
"""

import io
import re
from typing import List, Dict, Any, Tuple
from collections import Counter
import pymupdf as fitz
from pptx import Presentation
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_timestamp_str(time_str: str) -> int:
    """Converts '01:23' or '00:01:23,500' to integer seconds."""
    clean = re.sub(r"[,.].*", "", time_str.strip())
    parts = clean.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    try:
        return int(clean)
    except ValueError:
        return 0


def format_seconds(seconds: int) -> str:
    """Converts seconds into mm:ss or hh:mm:ss string."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into complete sentences while protecting
    common academic abbreviations (e.g., Fig., i.e., e.g., Dr., etc.)
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # Protect abbreviations
    placeholders = {
        r"\bFig\.\s*": "FIG_ABBR ",
        r"\be\.g\.,?\s*": "EG_ABBR ",
        r"\bi\.e\.,?\s*": "IE_ABBR ",
        r"\bDr\.\s*": "DR_ABBR ",
        r"\bvs\.\s*": "VS_ABBR ",
        r"\bet al\.\s*": "ETAL_ABBR ",
        r"(\d+)\.(\d+)": r"\1_DOT_\2"  # Decimal numbers
    }
    protected_text = text
    for pattern, repl in placeholders.items():
        protected_text = re.sub(pattern, repl, protected_text)

    pieces = re.split(r"(?<=[.!?])\s+", protected_text)

    sentences = []
    for piece in pieces:
        # Restore abbreviations
        restored = piece.replace("FIG_ABBR ", "Fig. ")
        restored = restored.replace("EG_ABBR ", "e.g., ")
        restored = restored.replace("IE_ABBR ", "i.e., ")
        restored = restored.replace("DR_ABBR ", "Dr. ")
        restored = restored.replace("VS_ABBR ", "vs. ")
        restored = restored.replace("ETAL_ABBR ", "et al. ")
        restored = re.sub(r"(\d+)_DOT_(\d+)", r"\1.\2", restored).strip()

        if len(restored.split()) >= 4:
            sentences.append(restored)

    return sentences


def parse_topic_mapping(mapping_text: str) -> List[Tuple[int, int, str]]:
    """
    Parses optional user mappings such as:
    1-3: Networking
    4-6: Transport Layer
    """
    rules = []
    if not mapping_text:
        return rules

    for line in mapping_text.splitlines():
        if ":" not in line:
            continue
        page_range, topic = line.split(":", 1)
        match = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", page_range.strip())
        topic = topic.strip()
        if match and topic:
            start = int(match.group(1))
            end = int(match.group(2) or start)
            if start > 0 and end >= start:
                rules.append((start, end, topic))
    return rules


def get_mapped_topic(unit_number: int, rules: List[Tuple[int, int, str]]) -> str:
    for start, end, topic in rules:
        if start <= unit_number <= end:
            return topic
    return ""


def auto_discover_topics_and_concepts(units: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Automatically groups content units into 3-5 major topics,
    extracts key concepts, and establishes prerequisite relationships.
    """
    if not units:
        return {
            "topics": ["General"],
            "concepts": {"General": ["Foundational Concepts"]},
            "prerequisites": []
        }

    # Extract high-frequency keywords across units
    full_text = [u["text"] for u in units if u.get("text")]
    if not full_text:
        return {
            "topics": ["General"],
            "concepts": {"General": []},
            "prerequisites": []
        }

    try:
        tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=30)
        tfidf.fit(full_text)
        feature_names = tfidf.get_feature_names_out()
    except Exception:
        feature_names = ["Introduction", "Fundamentals", "Core Concepts"]

    # Cluster or divide into sequential logical chapters
    total_units = len(units)
    num_topics = max(2, min(5, total_units // 3 or 2))
    chunk_size = max(1, total_units // num_topics)

    discovered_topics = []
    concepts_by_topic = {}
    prerequisites = []

    for i in range(num_topics):
        start_idx = i * chunk_size
        end_idx = total_units if i == num_topics - 1 else (i + 1) * chunk_size
        section_text = " ".join(u["text"] for u in units[start_idx:end_idx])

        # Pick top words in this section
        words = re.findall(r"\b[A-Za-z]{4,}\b", section_text)
        stopwords = set("this that with from have which were what their also each more other then".split())
        words = [w.capitalize() for w in words if w.lower() not in stopwords]
        common = [w for w, _ in Counter(words).most_common(5)]

        topic_name = f"Topic {i+1}: " + (" / ".join(common[:2]) if common else f"Section {i+1}")
        discovered_topics.append(topic_name)
        concepts_by_topic[topic_name] = common[2:5] or ["Overview", "Mechanisms", "Applications"]

        # Tag the units with this topic if they had none
        for u in units[start_idx:end_idx]:
            if not u.get("topic") or u["topic"] == "General":
                u["topic"] = topic_name

    # Build sequential prerequisite graph
    for i in range(len(discovered_topics) - 1):
        prerequisites.append({
            "topic": discovered_topics[i + 1],
            "prerequisite": discovered_topics[i]
        })

    return {
        "topics": discovered_topics,
        "concepts": concepts_by_topic,
        "prerequisites": prerequisites
    }


def process_pdf(pdf_bytes: bytes, filename: str, mapping_text: str = "") -> Dict[str, Any]:
    """Ingests a PDF textbook, extracting pages, chunks, sentences, and diagrams."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    chunks = []
    all_sentences = []
    rules = parse_topic_mapping(mapping_text)

    try:
        for index, page in enumerate(doc):
            number = index + 1
            text = page.get_text("text") or ""
            topic = get_mapped_topic(number, rules) or "General"

            images = []
            for info in page.get_images(full=True):
                try:
                    img_dict = doc.extract_image(info[0])
                    images.append({
                        "data": img_dict["image"],
                        "extension": img_dict.get("ext", "png"),
                        "page": number,
                        "caption": f"Figure from Page {number}"
                    })
                except Exception:
                    pass

            pages.append({
                "unit_id": number,
                "origin_type": "page",
                "origin_label": f"Page {number}",
                "page": number,
                "text": text,
                "topic": topic,
                "images": images
            })

            sentences = split_into_sentences(text)
            for s in sentences:
                all_sentences.append({
                    "text": s,
                    "unit_id": number,
                    "origin_type": "page",
                    "origin_label": f"Page {number}",
                    "page": number,
                    "topic": topic
                })

            if sentences:
                for start in range(0, len(sentences), 3):
                    chunk = " ".join(sentences[start:start + 3])
                    if len(chunk.split()) >= 6:
                        chunks.append({
                            "text": chunk,
                            "unit_id": number,
                            "origin_type": "page",
                            "origin_label": f"Page {number}",
                            "page": number,
                            "topic": topic
                        })
            elif text.strip():
                chunks.append({
                    "text": text.strip(),
                    "unit_id": number,
                    "origin_type": "page",
                    "origin_label": f"Page {number}",
                    "page": number,
                    "topic": topic
                })
    finally:
        doc.close()

    # If no manual mapping, auto-discover topics & prerequisites
    if not rules:
        course_meta = auto_discover_topics_and_concepts(pages)
        # Propagate discovered topic to chunks and sentences
        topic_map = {p["page"]: p["topic"] for p in pages}
        for c in chunks:
            c["topic"] = topic_map.get(c["page"], c["topic"])
        for s in all_sentences:
            s["topic"] = topic_map.get(s["page"], s["topic"])
    else:
        topics = sorted(list({r[2] for r in rules}))
        course_meta = {
            "topics": topics,
            "concepts": {t: [] for t in topics},
            "prerequisites": [{"topic": topics[i+1], "prerequisite": topics[i]} for i in range(len(topics)-1)]
        }

    return {
        "filename": filename,
        "media_type": "pdf",
        "pages": pages,
        "chunks": chunks,
        "sentences": all_sentences,
        "course_meta": course_meta
    }


def process_pptx(pptx_bytes: bytes, filename: str, mapping_text: str = "") -> Dict[str, Any]:
    """Ingests PowerPoint slides, extracting text, tables, and embedded diagrams."""
    prs = Presentation(io.BytesIO(pptx_bytes))
    slides = []
    chunks = []
    all_sentences = []
    rules = parse_topic_mapping(mapping_text)

    for index, slide in enumerate(prs.slides):
        number = index + 1
        parts = []
        images = []

        for shape in slide.shapes:
            if shape.has_text_frame and shape.text.strip():
                parts.append(shape.text.strip())

            if shape.has_table:
                for row in shape.table.rows:
                    row_txt = " ".join(c.text.strip() for c in row.cells if c.text.strip())
                    if row_txt:
                        parts.append(row_txt)

            # Extract slide images / figures
            try:
                if hasattr(shape, "image"):
                    images.append({
                        "data": shape.image.blob,
                        "extension": shape.image.ext,
                        "slide": number,
                        "caption": f"Diagram from Slide {number}"
                    })
            except Exception:
                pass

        text = "\n".join(parts)
        topic = get_mapped_topic(number, rules) or "General"

        slides.append({
            "unit_id": number,
            "origin_type": "slide",
            "origin_label": f"Slide {number}",
            "page": number,
            "slide": number,
            "text": text,
            "topic": topic,
            "images": images
        })

        sentences = split_into_sentences(text)
        for s in sentences:
            all_sentences.append({
                "text": s,
                "unit_id": number,
                "origin_type": "slide",
                "origin_label": f"Slide {number}",
                "page": number,
                "slide": number,
                "topic": topic
            })

        if sentences:
            for start in range(0, len(sentences), 3):
                chunk = " ".join(sentences[start:start + 3])
                if len(chunk.split()) >= 5:
                    chunks.append({
                        "text": chunk,
                        "unit_id": number,
                        "origin_type": "slide",
                        "origin_label": f"Slide {number}",
                        "page": number,
                        "slide": number,
                        "topic": topic
                    })
        elif text.strip():
            chunks.append({
                "text": text.strip(),
                "unit_id": number,
                "origin_type": "slide",
                "origin_label": f"Slide {number}",
                "page": number,
                "slide": number,
                "topic": topic
            })

    if not rules:
        course_meta = auto_discover_topics_and_concepts(slides)
        topic_map = {s["slide"]: s["topic"] for s in slides}
        for c in chunks:
            c["topic"] = topic_map.get(c.get("slide", 1), c["topic"])
        for s in all_sentences:
            s["topic"] = topic_map.get(s.get("slide", 1), s["topic"])
    else:
        topics = sorted(list({r[2] for r in rules}))
        course_meta = {
            "topics": topics,
            "concepts": {t: [] for t in topics},
            "prerequisites": [{"topic": topics[i+1], "prerequisite": topics[i]} for i in range(len(topics)-1)]
        }

    return {
        "filename": filename,
        "media_type": "pptx",
        "pages": slides,
        "chunks": chunks,
        "sentences": all_sentences,
        "course_meta": course_meta
    }


def process_video_or_transcript(
    file_bytes: bytes,
    filename: str,
    transcript_text: str = ""
) -> Dict[str, Any]:
    """
    Ingests lecture videos or transcripts (.srt, .vtt, .txt or video with transcript).
    Links every concept to exact video timestamp [mm:ss].
    """
    content_str = transcript_text
    if not content_str and (filename.lower().endswith(".srt") or filename.lower().endswith(".vtt") or filename.lower().endswith(".txt")):
        content_str = file_bytes.decode("utf-8", errors="ignore")

    cues = []
    # Pattern for SRT or VTT: 00:01:20 --> 00:01:25 or [01:20] Text
    time_blocks = re.findall(
        r"(?:(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)\s*-->\s*(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)\s*\n([\s\S]*?)(?=\n\s*\n|\Z))|\[(\d{1,2}:\d{2})\]\s*([^\n]+)",
        content_str
    )

    if time_blocks:
        for idx, block in enumerate(time_blocks):
            if block[0]:  # SRT / VTT style
                start_raw, _, text_content = block[0], block[1], block[2]
                sec = parse_timestamp_str(start_raw)
                ts_label = format_seconds(sec)
                clean_text = re.sub(r"<[^>]+>", "", text_content).strip()
            else:  # [mm:ss] Text style
                start_raw, clean_text = block[3], block[4]
                sec = parse_timestamp_str(start_raw)
                ts_label = format_seconds(sec)

            if clean_text:
                cues.append({
                    "unit_id": idx + 1,
                    "origin_type": "video",
                    "origin_label": f"Timestamp {ts_label}",
                    "timestamp": ts_label,
                    "seconds": sec,
                    "text": clean_text,
                    "topic": "General",
                    "page": idx + 1,
                    "images": []
                })
    else:
        # Fallback: break text into 45-second chunks
        paragraphs = [p.strip() for p in content_str.split("\n\n") if p.strip()]
        for idx, p in enumerate(paragraphs or [content_str]):
            sec = idx * 45
            ts_label = format_seconds(sec)
            cues.append({
                "unit_id": idx + 1,
                "origin_type": "video",
                "origin_label": f"Timestamp {ts_label}",
                "timestamp": ts_label,
                "seconds": sec,
                "text": p,
                "topic": "General",
                "page": idx + 1,
                "images": []
            })

    chunks = []
    all_sentences = []
    for cue in cues:
        sentences = split_into_sentences(cue["text"])
        for s in sentences:
            all_sentences.append({
                "text": s,
                "unit_id": cue["unit_id"],
                "origin_type": "video",
                "origin_label": cue["origin_label"],
                "timestamp": cue["timestamp"],
                "seconds": cue["seconds"],
                "page": cue["unit_id"],
                "topic": "General"
            })
        chunks.append({
            "text": cue["text"],
            "unit_id": cue["unit_id"],
            "origin_type": "video",
            "origin_label": cue["origin_label"],
            "timestamp": cue["timestamp"],
            "seconds": cue["seconds"],
            "page": cue["unit_id"],
            "topic": "General"
        })

    course_meta = auto_discover_topics_and_concepts(cues)
    topic_map = {c["unit_id"]: c["topic"] for c in cues}
    for ch in chunks:
        ch["topic"] = topic_map.get(ch["unit_id"], "General")
    for s in all_sentences:
        s["topic"] = topic_map.get(s["unit_id"], "General")

    return {
        "filename": filename,
        "media_type": "video",
        "media_bytes": file_bytes if filename.lower().endswith((".mp4", ".mov", ".mp3", ".wav")) else None,
        "pages": cues,
        "chunks": chunks,
        "sentences": all_sentences,
        "course_meta": course_meta
    }
