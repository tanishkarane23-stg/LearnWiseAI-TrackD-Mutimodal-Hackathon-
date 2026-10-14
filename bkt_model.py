"""
bkt_model.py
============
Bayesian Knowledge Tracing (BKT) and Learner State Persistence
Part of LearnWise AI - Track D: Personalized Tutoring & Adaptive Learning.

This module models a student's evolving topic mastery using standard Bayesian
Knowledge Tracing formulas and persists the learner profile locally.
"""

import json
import os
import time
from typing import Dict, Any, List

PROFILE_FILE = os.path.join(os.path.dirname(__file__), "learner_profile.json")

# Standard BKT Default Parameters
DEFAULT_P_L0 = 0.25   # Prior probability of knowing a topic
DEFAULT_P_TRANSIT = 0.15  # Probability of learning during an opportunity
DEFAULT_P_SLIP = 0.10     # Probability of making an accidental mistake despite mastery
DEFAULT_P_GUESS_MCQ = 0.20  # Chance of guessing correctly on 4-option MCQ
DEFAULT_P_GUESS_OPEN = 0.05  # Chance of guessing short answer or numerical


class BayesianKnowledgeTracer:
    """
    Maintains per-topic mastery probabilities P(L) using standard BKT formulas.
    """

    def __init__(self, p_transit=DEFAULT_P_TRANSIT, p_slip=DEFAULT_P_SLIP):
        self.p_transit = p_transit
        self.p_slip = p_slip

    def update_mastery(
        self,
        current_p_l: float,
        is_correct: bool,
        question_type: str = "MCQ"
    ) -> float:
        """
        Updates the topic mastery probability P(L) given a student's answer.
        """
        # Select appropriate guess parameter
        p_guess = DEFAULT_P_GUESS_MCQ if question_type == "MCQ" else DEFAULT_P_GUESS_OPEN

        # Ensure current probability is strictly within (0, 1)
        p_l = min(max(float(current_p_l), 0.01), 0.99)

        # 1. Bayesian posterior calculation P(L_t | observation)
        if is_correct:
            numerator = p_l * (1.0 - self.p_slip)
            denominator = numerator + ((1.0 - p_l) * p_guess)
        else:
            numerator = p_l * self.p_slip
            denominator = numerator + ((1.0 - p_l) * (1.0 - p_guess))

        p_l_given_obs = numerator / max(denominator, 1e-9)

        # 2. Transition update: probability of learning during this opportunity
        p_l_next = p_l_given_obs + ((1.0 - p_l_given_obs) * self.p_transit)

        # Clamp between 1% and 99% for numerical stability
        return round(min(max(p_l_next, 0.01), 0.99), 4)

    def update_from_conversation(self, current_p_l: float) -> float:
        """
        Gives a gentle mastery reinforcement when a student actively studies
        or asks grounded questions about a topic in the tutor chat.
        """
        p_l = min(max(float(current_p_l), 0.01), 0.99)
        # Half the standard transition rate for active reading
        p_l_next = p_l + ((1.0 - p_l) * (self.p_transit * 0.35))
        return round(min(max(p_l_next, 0.01), 0.99), 4)


class LearnerProfileManager:
    """
    Handles student profile state, history logging, and local JSON persistence.
    """

    def __init__(self, filepath: str = PROFILE_FILE):
        self.filepath = filepath
        self.bkt = BayesianKnowledgeTracer()
        self.profile = self.load_profile()

    def get_default_profile(self) -> Dict[str, Any]:
        return {
            "student_id": "student_default",
            "experience_level": "Beginner",
            "diagnostic_completed": False,
            "topic_mastery": {},  # { topic_name: { "p_l": float, "attempts": int, "correct": int, "history": [] } }
            "quiz_attempts": [],
            "conversation_count": 0,
            "created_at": time.time(),
            "last_active": time.time()
        }

    def load_profile(self) -> Dict[str, Any]:
        """Loads learner profile from disk, or initializes defaults."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Merge any missing default keys
                    defaults = self.get_default_profile()
                    for k, v in defaults.items():
                        if k not in data:
                            data[k] = v
                    return data
            except Exception:
                pass
        return self.get_default_profile()

    def save_profile(self):
        """Persists learner profile to disk."""
        self.profile["last_active"] = time.time()
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.profile, f, indent=2)
        except Exception:
            pass

    def get_topic_mastery(self, topic: str) -> float:
        """Returns the BKT mastery probability P(L) for a topic."""
        topic_data = self.profile["topic_mastery"].get(topic)
        if topic_data:
            return topic_data.get("p_l", DEFAULT_P_L0)
        return DEFAULT_P_L0

    def get_all_mastery(self) -> Dict[str, float]:
        """Returns {topic: p_l} mapping."""
        return {
            topic: data.get("p_l", DEFAULT_P_L0)
            for topic, data in self.profile["topic_mastery"].items()
        }

    def set_intake_level(self, level: str):
        """
        Calibrates initial prior P(L_0) based on student intake selection.
        """
        p_l0_map = {
            "Beginner": 0.15,
            "Intermediate": 0.40,
            "Advanced": 0.70
        }
        prior = p_l0_map.get(level, DEFAULT_P_L0)
        self.profile["experience_level"] = level
        self.profile["diagnostic_completed"] = True

        for topic in self.profile["topic_mastery"]:
            if self.profile["topic_mastery"][topic]["attempts"] == 0:
                self.profile["topic_mastery"][topic]["p_l"] = prior
        self.save_profile()

    def record_question_result(
        self,
        topic: str,
        is_correct: bool,
        question_type: str = "MCQ"
    ) -> float:
        """
        Updates BKT mastery after a question attempt.
        """
        if topic not in self.profile["topic_mastery"]:
            prior = 0.15 if self.profile["experience_level"] == "Beginner" else DEFAULT_P_L0
            self.profile["topic_mastery"][topic] = {
                "p_l": prior,
                "attempts": 0,
                "correct": 0,
                "history": []
            }

        topic_data = self.profile["topic_mastery"][topic]
        old_p_l = topic_data["p_l"]
        new_p_l = self.bkt.update_mastery(old_p_l, is_correct, question_type)

        topic_data["p_l"] = new_p_l
        topic_data["attempts"] += 1
        if is_correct:
            topic_data["correct"] += 1

        topic_data["history"].append({
            "timestamp": time.time(),
            "is_correct": is_correct,
            "p_l_after": new_p_l,
            "question_type": question_type
        })

        self.save_profile()
        return new_p_l

    def record_conversation_engagement(self, topic: str) -> float:
        """
        Gives a mild reinforcement to topic familiarity when chatting with the tutor.
        """
        if not topic or topic == "All topics" or topic == "General":
            return DEFAULT_P_L0

        if topic not in self.profile["topic_mastery"]:
            self.profile["topic_mastery"][topic] = {
                "p_l": DEFAULT_P_L0,
                "attempts": 0,
                "correct": 0,
                "history": []
            }

        topic_data = self.profile["topic_mastery"][topic]
        new_p_l = self.bkt.update_from_conversation(topic_data["p_l"])
        topic_data["p_l"] = new_p_l
        self.profile["conversation_count"] = self.profile.get("conversation_count", 0) + 1
        self.save_profile()
        return new_p_l

    def record_quiz_summary(
        self,
        topic: str,
        correct_count: int,
        total_questions: int,
        misconceptions: List[str] = None
    ):
        """Records an entire quiz attempt in the student history."""
        self.profile["quiz_attempts"].append({
            "timestamp": time.time(),
            "topic": topic,
            "correct": correct_count,
            "total": total_questions,
            "accuracy": (correct_count / total_questions) * 100 if total_questions else 0.0,
            "misconceptions": misconceptions or []
        })
        self.save_profile()

    def reset_profile(self):
        """Resets the profile to initial state."""
        self.profile = self.get_default_profile()
        self.save_profile()
