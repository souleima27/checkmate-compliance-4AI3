import os
import json
from datetime import datetime
from typing import Dict, Any, List

FEEDBACK_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workflow", "output", "feedback.json")

class FeedbackManager:
    def __init__(self, feedback_file: str = FEEDBACK_FILE):
        self.feedback_file = feedback_file
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not os.path.exists(os.path.dirname(self.feedback_file)):
            os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump([], f)

    def add_feedback(self, feedback_data: Dict[str, Any]):
        """
        Add a feedback entry.
        feedback_data should contain:
        - type: "chatbot" | "violation"
        - id: string (message id or violation id)
        - feedback: "like" | "dislike"
        - details: dict (optional context like rule_id, agent, etc.)
        """
        try:
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = []

        entry = {
            "timestamp": datetime.now().isoformat(),
            **feedback_data
        }
        
        data.append(entry)

        with open(self.feedback_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return entry

    def get_all_feedback(self) -> List[Dict[str, Any]]:
        try:
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
