
import io
import re
import random
from collections import Counter

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="LearnWise AI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 LearnWise AI")
st.caption(
    "Your source-grounded study companion | "
    "Learn from your own course materials"
)

STOP_WORDS = set("""
a an the and or but if then than so because as of at by for from
in into on onto to with without is am are was were be been being
it its this that these those they them their he she we you i my
our your have has had do does did can could should would will
may might not no yes which who whom what when where why how
all any each every both few more most other some such only same
also very just about over under between through during before after
use used using based according include includes including example
examples figure table chapter section page method methods result
results one two three first second third new different important
following called known make made many much several type types
"""
.split())


# --------------------------------------------------
# 2. SESSION STATE
# --------------------------------------------------

defaults = {
    "pages": [],
    "chunks": [],
    "sentences": [],
    "document_name": "",
    "document_ready": False,
    "chat_history": [],
    "quiz": [],
    "quiz_topic": "",
    "quiz_submitted": False,
    "quiz_score": None,
    "mastery": {},
    "attempt_history": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# 3. TEXT PROCESSING
# --------------------------------------------------

def split_sentences(text):
    """Simple sentence splitting for a first prototype."""
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+", text)

    return [
        part.strip()
        for part in parts
        if len(part.split()) >= 6
    ]


def parse_topic_mapping(mapping_text):
    """
    Example:
    1-3: Biology
    4-6: Genetics
    """
    rules = []

    for line in mapping_text.splitlines():
        if ":" not in line:
            continue

        page_range, topic = line.split(":", 1)
        topic = topic.strip()

        if not topic:
            continue

        match = re.fullmatch(
            r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?",
            page_range.strip()
        )

        if not match:
            continue

        start_page = int(match.group(1))
        end_page = int(match.group(2) or start_page)

        if start_page > 0 and end_page >= start_page:
            rules.append((start_page, end_page, topic))

    return rules


def get_topic(page_number, rules):
    for start_page, end_page, topic in rules:
        if start_page <= page_number <= end_page:
            return topic

    return "General"


def process_pdf(pdf_bytes, filename, mapping_text):
    """Extract pages and build source-linked text chunks."""
    document = fitz.open(
    stream=pdf_bytes,        
    filetype="pdf"
    )

    pages = []
    chunks = []
    all_sentences = []

    rules = parse_topic_mapping(mapping_text)

    for index, page in enumerate(document):
        page_number = index + 1
        text = page.get_text("text")
        topic = get_topic(page_number, rules)

        pages.append({
            "page": page_number,
            "text": text,
            "topic": topic
        })

        sentences = split_sentences(text)

        for sentence in sentences:
            all_sentences.append({
                "text": sentence,
                "page": page_number,
                "topic": topic
            })

        # Three-sentence chunks provide more context for retrieval.
        if sentences:
            for start in range(0, len(sentences), 3):
                chunk_text = " ".join(
                    sentences[start:start + 3]
                )

                if len(chunk_text.split()) >= 6:
                    chunks.append({
                        "text": chunk_text,
                        "page": page_number,
                        "topic": topic
                    })
        elif text.strip():
            # Fallback for pages that do not split into sentences.
            chunks.append({
                "text": text.strip(),
                "page": page_number,
                "topic": topic
            })

    document.close()

    if not any(p["text"].strip() for p in pages):
        raise ValueError(
            "No selectable text was found. "
            "This may be a scanned PDF; OCR is not included "
            "in this first version."
        )

    st.session_state.pages = pages
    st.session_state.chunks = chunks
    st.session_state.sentences = all_sentences
    st.session_state.document_name = filename
    st.session_state.document_ready = True
    st.session_state.chat_history = []
    st.session_state.quiz = []
    st.session_state.quiz_submitted = False
    st.session_state.quiz_score = None


# --------------------------------------------------
# 4. RETRIEVAL
# --------------------------------------------------

def retrieve_passages(question, topic=None, top_k=4):

 chunks = st.session_state.chunks

 if topic and topic != "All topics":
    chunks = [
        chunk for chunk in chunks
        if chunk["topic"] == topic
    ]

 if not chunks:
    return []

 # Remove common words to focus on meaningful terms.
 question_words = set(
    re.findall(r"\b[a-zA-Z]{2,}\b", question.lower())
 )

 question_terms = question_words - STOP_WORDS

 if not question_terms:
    return []

 texts = [chunk["text"] for chunk in chunks]

 try:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        query_vector,
        matrix
    ).flatten()

 except ValueError:
    return []

 # Rank passages by TF-IDF similarity.
 ranked_indices = scores.argsort()[::-1]

 results = []

 for index in ranked_indices:
    chunk = chunks[index]
    score = float(scores[index])

    passage_words = set(
        re.findall(
            r"\b[a-zA-Z]{2,}\b",
            chunk["text"].lower()
        )
    )

    passage_terms = passage_words - STOP_WORDS

    # Count meaningful question terms found in the passage.
    matched_terms = question_terms.intersection(
        passage_terms
    )

    overlap_count = len(matched_terms)

    # Ignore passages with no meaningful lexical overlap.
    if overlap_count == 0:
        continue

    results.append({
        **chunk,
        "similarity": score,
        "overlap_count": overlap_count,
        "matched_terms": sorted(matched_terms)
    })

    if len(results) >= top_k:
        break

 return results


def answer_question(question, topic):
 #Return relevant source evidence or refuse unsupported questions."""
 
 if len(question.strip()) < 3:
    return {
        "supported": False,
        "message": "Please enter a more specific question.",
        "sources": []
    }

 results = retrieve_passages(
    question,
    topic=topic,
    top_k=4
 )

 if not results:
    return {
        "supported": False,
        "message": (
            "I couldn't find matching evidence in your study "
            "material. Try using terms from your notes or "
            "rephrasing the question."
        ),
        "sources": []
    }

 best_match = results[0]

 # These are initial heuristic thresholds, not validated
 # measures of answer correctness.
 enough_similarity = best_match["similarity"] >= 0.10

 question_word_count = len(
    set(
        re.findall(
            r"\b[a-zA-Z]{2,}\b",
            question.lower()
        )
    ) - STOP_WORDS
 )

 enough_overlap = (
    best_match["overlap_count"] >=
    min(2, question_word_count)
 )

 if not (enough_similarity and enough_overlap):
    return {
        "supported": False,
        "message": (
            "I couldn't find sufficiently relevant evidence "
            "to support an answer from your study material. "
            "I won't guess. Try rephrasing your question "
            "or selecting another topic."
        ),
        "sources": []
    }

 # Only show passages that meet the initial relevance threshold.
 relevant_sources = [
    source for source in results
    if source["similarity"] >= 0.10
 ]

 if not relevant_sources:
    return {
        "supported": False,
        "message": (
            "The retrieved passages were not relevant enough "
            "to support an answer. Please try rephrasing "
            "your question."
        ),
        "sources": []
    }

 return {
    "supported": True,
    "message": (
        "I found the following evidence in your study material. "
        "Read the excerpts together for a fuller context. "
        "The passages are retrieved directly from your PDF; "
        "they are not a newly generated explanation. "
        "Check the cited pages to verify the answer."
    ),
    "sources": relevant_sources
 }


# --------------------------------------------------
# 5. QUIZ GENERATION
# --------------------------------------------------

def get_content_words(text):
    words = re.findall(r"\b[A-Za-z][A-Za-z-]{4,}\b", text)

    return [
        word for word in words
        if word.lower() not in STOP_WORDS
        and not word.isupper()
    ]


def generate_quiz(topic, number_of_questions=5):
# Generate source-linked fill-in-the-blank questions.

 sentences = [
    item for item in st.session_state.sentences
    if topic == "All topics" or item["topic"] == topic
 ]

 # Exclude sentences that are likely to contain book metadata.
 metadata_phrases = [
    "isbn",
    "all rights reserved",
    "printed in",
    "published by",
    "publisher",
    "publication address",
    "copyright",
    "first edition",
    "second edition",
    "book objectives",
    "objectives of this book",
    "about the author",
    "table of contents",
    "acknowledgements",
    "acknowledgments",
    "preface",
    "foreword"
 ]

 sentences = [
    item for item in sentences
    if not any(
        phrase in item["text"].lower()
        for phrase in metadata_phrases
    )
 ] 

 if not sentences:
    return []

 all_words = []

 for item in sentences:
    all_words.extend(get_content_words(item["text"]))

 frequencies = Counter(
    word.lower() for word in all_words
 )

 display_words = {}

 for word in all_words:
    display_words.setdefault(word.lower(), word)

 candidate_terms = sorted(
    display_words.keys(),
    key=lambda word: (-frequencies[word], word)
 )

 questions = []
 used_sentences = set()

 for item in sentences:
    sentence = item["text"]
    sentence_key = sentence.lower()

    if sentence_key in used_sentences:
        continue

    if len(sentence.split()) < 8:
        continue

    candidates = get_content_words(sentence)

    if len(candidates) < 2:
        continue

    # Prefer words that are meaningful but not always
    # the longest word in the sentence.
    suitable_answers = [
        word for word in candidates
        if len(word) >= 5
    ]

    if not suitable_answers:
        continue

    # Use a stable sentence-based random generator so the
    # selected answer is reproducible for the same sentence.
    rng = random.Random(sentence_key)

    answer = rng.choice(suitable_answers)
    answer_lower = answer.lower()

    # Exclude words already present in the sentence.
    other_terms = [
        term for term in candidate_terms
        if term != answer_lower
        and not re.search(
            r"\b" + re.escape(term) + r"\b",
            sentence.lower()
        )
    ]

    # Prefer distractors of a similar length.
    similar_terms = [
        term for term in other_terms
        if abs(len(term) - len(answer)) <= 3
    ]

    if len(similar_terms) >= 3:
        distractor_pool = similar_terms
    else:
        distractor_pool = other_terms

    if len(distractor_pool) < 3:
        continue

    # Shuffle the distractors before choosing three.
    rng.shuffle(distractor_pool)

    distractors = [
        display_words[term]
        for term in distractor_pool[:3]
    ]

    # Replace only the selected answer word.
    question_text = re.sub(
        r"\b" + re.escape(answer) + r"\b",
        "________",
        sentence,
        count=1,
        flags=re.IGNORECASE
    )

    options = [answer] + distractors

    # Avoid always displaying the correct answer first.
    rng.shuffle(options)

    questions.append({
        "question": question_text,
        "options": options,
        "answer": answer,
        "page": item["page"],
        "topic": item["topic"],
        "source_sentence": sentence
    })

    used_sentences.add(sentence_key)

    if len(questions) >= number_of_questions:
        break

 return questions


# --------------------------------------------------
# 6. LEARNER MODEL
# --------------------------------------------------

def record_quiz_result(topic, correct, total):
    mastery = st.session_state.mastery

    if topic not in mastery:
        mastery[topic] = {
            "correct": 0,
            "attempted": 0
        }

    mastery[topic]["correct"] += correct
    mastery[topic]["attempted"] += total

    st.session_state.attempt_history.append({
        "topic": topic,
        "correct": correct,
        "total": total
    })


def mastery_percentage(topic):
    record = st.session_state.mastery.get(topic)

    if not record or record["attempted"] == 0:
        return 0.0

    return (
        record["correct"] / record["attempted"]
    ) * 100


# --------------------------------------------------
# 7. SIDEBAR: UPLOAD MATERIAL
# --------------------------------------------------

with st.sidebar:
    st.header("📄 Study material")

    uploaded_file = st.file_uploader(
        "Upload a text-based PDF",
        type=["pdf"]
    )

    st.caption(
        "Scanned PDFs and image-only slides need OCR, "
        "which is not included in this first version."
    )

    st.subheader("Optional topic mapping")

    mapping_text = st.text_area(
        "Assign topics to page ranges",
        placeholder=(
            "1-3: Biology\n"
            "4-6: Genetics\n"
            "7: Evolution"
        ),
        help=(
            "Use one rule per line. Pages without a rule "
            "are assigned to General."
        )
    )

    if st.button("Process PDF", type="primary"):
        if uploaded_file is None:
            st.error("Please upload a PDF first.")
        else:
            try:
                with st.spinner("Extracting PDF text..."):
                    process_pdf(
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                        mapping_text
                    )

                st.success("PDF processed successfully!")

            except Exception as error:
                st.error(f"Could not process the PDF: {error}")

    if st.session_state.document_ready:
        st.divider()
        st.success(
            f"Loaded: {st.session_state.document_name}"
        )

        st.write(
            f"Pages: {len(st.session_state.pages)}"
        )
        st.write(
            f"Text chunks: {len(st.session_state.chunks)}"
        )
        st.write(
            f"Quiz sentences: {len(st.session_state.sentences)}"
        )


# --------------------------------------------------
# 8. MAIN NAVIGATION
# --------------------------------------------------

tab_tutor, tab_quiz, tab_dashboard = st.tabs([
    "💬 Tutor",
    "📝 Adaptive quiz",
    "📊 Dashboard"
])

if not st.session_state.document_ready:
    st.info(
        "Start by uploading a text-based PDF in the sidebar "
        "and clicking **Process PDF**."
    )

else:
    topics = sorted({
        chunk["topic"]
        for chunk in st.session_state.chunks
    })

    if not topics:
        topics = ["General"]

    with tab_tutor:
        st.header("Ask your study material")

        tutor_topic = st.selectbox(
            "Search within",
            ["All topics"] + topics,
            key="tutor_topic"
        )

        st.caption(
            "Answers are based on retrieved excerpts, not "
            "independent medical, legal, or other expert advice."
        )

        with st.form("tutor_form", clear_on_submit=True):
            question = st.text_input(
                "What would you like to understand?",
                placeholder="Ask a question about your PDF..."
            )

            ask_button = st.form_submit_button("Find evidence")

        if ask_button:
            response = answer_question(question, tutor_topic)

            st.session_state.chat_history.append({
                "question": question,
                "response": response
            })

        for chat in reversed(st.session_state.chat_history):
            st.markdown("---")
            st.markdown(f"**You asked:** {chat['question']}")
            st.write(chat["response"]["message"])

            if chat["response"]["supported"]:
                for source in chat["response"]["sources"]:
                    with st.expander(
                        f"Source: page {source['page']} "
                        f"• {source['topic']} "
                        f"• relevance {source['similarity']:.2f}"
                    ):
                        st.write(source["text"])
                        st.caption(
                         "Matched question terms: "
                         + ", ".join(source["matched_terms"])
                        )
                        

                        st.caption(
                            f"Source: {st.session_state.document_name}, "
                            f"page {source['page']}. "
                            "Relevance score is not a truth score."
                        )

            elif chat["response"]["sources"]:
                st.caption("Low-relevance results (not treated as evidence):")

                for source in chat["response"]["sources"]:
                    st.write(
                        f"Page {source['page']}: {source['text']}"
                    )

    with tab_quiz:
        st.header("Practice and improve")

        if st.session_state.mastery:
            weakest_topic = min(
                st.session_state.mastery.keys(),
                key=mastery_percentage
            )

            st.info(
                f"Recommended revision topic: **{weakest_topic}**"
            )
        else:
            weakest_topic = topics[0]

            st.info(
                "Complete your first quiz to establish a baseline "
                "for topic mastery."
            )

        default_index = (
            topics.index(weakest_topic)
            if weakest_topic in topics
            else 0
        )

        quiz_topic = st.selectbox(
            "Choose a quiz topic",
            topics,
            index=default_index,
            key="quiz_topic_selector"
        )

        if st.button("Generate 5-question quiz"):
            new_quiz = generate_quiz(
                quiz_topic,
                number_of_questions=5
            )

            st.session_state.quiz = new_quiz
            st.session_state.quiz_topic = quiz_topic
            st.session_state.quiz_submitted = False
            st.session_state.quiz_score = None

            if not new_quiz:
                st.warning(
                    "Not enough suitable sentences and distractor "
                    "terms were found. Try a different topic or "
                    "a longer PDF."
                )
            else:
                st.rerun()

        if st.session_state.quiz:
            st.write(
                f"Quiz topic: **{st.session_state.quiz_topic}**"
            )

            if not st.session_state.quiz_submitted:
                with st.form("quiz_form"):
                    for index, item in enumerate(
                        st.session_state.quiz
                    ):
                        st.markdown(
                            f"**Q{index + 1}.** {item['question']}"
                        )

                        st.radio(
                            "Choose one answer",
                            item["options"],
                            key=f"quiz_answer_{index}",
                            index=None
                        )

                    submitted = st.form_submit_button(
                        "Submit quiz",
                        type="primary"
                    )

                if submitted:
                    unanswered = [
                        index for index in range(
                            len(st.session_state.quiz)
                        )
                        if st.session_state.get(
                            f"quiz_answer_{index}"
                        ) is None
                    ]

                    if unanswered:
                        st.warning(
                            "Please answer every question before "
                            "submitting."
                        )
                    else:
                        correct_count = 0

                        for index, item in enumerate(
                            st.session_state.quiz
                        ):
                            selected = st.session_state.get(
                                f"quiz_answer_{index}"
                            )

                            is_correct = (
                                selected.lower()
                                == item["answer"].lower()
                            )

                            if is_correct:
                                correct_count += 1

                        record_quiz_result(
                            st.session_state.quiz_topic,
                            correct_count,
                            len(st.session_state.quiz)
                        )

                        st.session_state.quiz_score = (
                            correct_count,
                            len(st.session_state.quiz)
                        )

                        st.session_state.quiz_submitted = True
                        st.rerun()

            else:
                correct_count, total = (
                    st.session_state.quiz_score
                )

                st.metric(
                    "Your score",
                    f"{correct_count}/{total}",
                    f"{(correct_count / total) * 100:.0f}%"
                )

                for index, item in enumerate(
                    st.session_state.quiz
                ):
                    selected = st.session_state.get(
                        f"quiz_answer_{index}"
                    )

                    correct = (
                        selected.lower()
                        == item["answer"].lower()
                    )

                    if correct:
                        st.success(
                            f"Q{index + 1}: Correct — "
                            f"{item['answer']}"
                        )
                    else:
                        st.error(
                            f"Q{index + 1}: Your answer: {selected}. "
                            f"Correct answer: {item['answer']}."
                        )

                    st.caption(
                        f"Evidence: page {item['page']} — "
                        f"{item['source_sentence']}"
                    )

                if st.button("Create another quiz"):
                    st.session_state.quiz = []
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_score = None
                    st.rerun()

    with tab_dashboard:
        st.header("Your learning dashboard")

        total_attempted = sum(
            item["attempted"]
            for item in st.session_state.mastery.values()
        )

        total_correct = sum(
            item["correct"]
            for item in st.session_state.mastery.values()
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Topics available",
            len(topics)
        )

        col2.metric(
            "Questions attempted",
            total_attempted
        )

        overall_accuracy = (
            total_correct / total_attempted * 100
            if total_attempted else 0
        )

        col3.metric(
            "Overall quiz accuracy",
            f"{overall_accuracy:.1f}%"
        )

        dashboard_data = []

        for topic in topics:
            record = st.session_state.mastery.get(
                topic,
                {"correct": 0, "attempted": 0}
            )

            dashboard_data.append({
                "Topic": topic,
                "Mastery (%)": (
                    record["correct"] / record["attempted"] * 100
                    if record["attempted"] else 0
                ),
                "Attempts": record["attempted"]
            })

        df = pd.DataFrame(dashboard_data)

        st.subheader("Topic mastery estimates")

        fig = px.bar(
            df,
            x="Topic",
            y="Mastery (%)",
            color="Mastery (%)",
            range_y=[0, 100],
            text_auto=".1f",
            title="Accuracy-based mastery estimate"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "This initial mastery estimate is cumulative quiz "
            "accuracy, not a validated measure of actual learning."
        )

        st.subheader("Topic performance")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        if st.session_state.attempt_history:
            st.subheader("Quiz history")

            history_df = pd.DataFrame(
                st.session_state.attempt_history
            )

            history_df["Accuracy (%)"] = (
                history_df["correct"]
                / history_df["total"]
                * 100
            )

            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info(
                "Your dashboard will populate after your first quiz."
            )


# --------------------------------------------------
# 9. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "LearnWise AI prototype • Educational use only. "
    "Retrieved passages and automatically generated quiz questions "
    "may be incomplete or misleading. Verify them against the "
    "original course material."
)
