# LearnWise AI 

LearnWise AI is a multimodal, source-grounded study companion and adaptive tutor. It ingests your course materials (PDFs, PowerPoints, and video transcripts) and uses a local AI model to provide verified answers, generate adaptive quizzes, and track your topic mastery using Bayesian Knowledge Tracing (BKT).

##  Key Features

*   **Multimodal Knowledge Base:** Upload PDF textbooks, PPTX slides, or SRT/VTT/TXT video transcripts. The system automatically extracts text, diagrams, and semantic topics.
*   **Source-Grounded AI Tutor:** Ask questions in English, Hindi, or Hinglish. Every answer is strictly cited with the exact page, slide, or video timestamp to eliminate AI hallucinations.
*   **Adaptive Assessments:** Automatically generates MCQs, Short Answer, and Numerical questions based on your weakest topics. AI grading includes misconception detection.
*   **Bayesian Knowledge Tracing (BKT):** Tracks your estimated mastery probability (P(L)) for each topic in real-time.
*   **Revision Toolkit:** Auto-generates active-recall flashcards, a visual course prerequisite map, and a spaced-repetition study schedule.
*   **Local & Private:** Uses Ollama to run the LLM entirely on your local machine.

## Prerequisites

Before running the application, you must install and run **Ollama** locally, as LearnWise AI relies on it for generating quizzes and tutor responses.

1.  Download and install [Ollama](https://ollama.com/) for your operating system.
2.  Open your terminal or command prompt.
3.  Download and run the required `llama3.2` model by executing the following command:
    ```bash
    ollama run llama3.2
    ```
4.  Keep the Ollama terminal window running in the background while you use LearnWise AI.

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/learnwise-ai.git](https://github.com/yourusername/learnwise-ai.git)
    cd learnwise-ai
    ```

2.  **Set up a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Install all required Python libraries from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## Running the App

Once Ollama is running and your dependencies are installed, you can launch the LearnWise AI dashboard.

1.  Ensure your Ollama local server is active.
2.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
3.  The application will open automatically in your default web browser (usually at `http://localhost:8501`).

*(Note: If you need to revert to previous prototype builds, you can run `streamlit run app_backup.py` or `streamlit run app_before_full_upgrade.py`)*

## Project Structure

*   `app.py`: The main Streamlit application featuring the updated premium UI and multimodal tabs.
*   `bkt_model.py`: Contains the Bayesian Knowledge Tracing logic and learner profile JSON persistence.
*   `multimodal_loader.py`: Handles the ingestion, parsing, and chunking of PDFs, PPTXs, and video transcripts.
*   `evaluator.py`: Benchmarking suite for evaluating RAG faithfulness, context precision, and simulated student mastery trajectories.
*   `learner_profile.json`: Local storage file for tracking your ongoing BKT mastery progress and quiz history.
*   `requirements.txt`: Package dependencies.
