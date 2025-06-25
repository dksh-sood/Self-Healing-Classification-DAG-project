"""
LangGraph DAG nodes for self-healing classification
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, TypedDict, Optional

from model_wrapper import SentimentClassifier, BackupClassifier

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- State Definition ----
class GraphState(TypedDict):
    text: str
    predicted_label: Optional[str]
    confidence: Optional[float]
    full_results: Optional[Dict]
    needs_fallback: bool
    fallback_activated: bool
    user_feedback: Optional[str]
    final_label: Optional[str]
    method_used: str
    timestamp: str
    confidence_threshold: float


# ---- Inference Node ----
class InferenceNode:
    def __init__(self, model_path: str):
        self.classifier = SentimentClassifier(model_path)

    def __call__(self, state: GraphState) -> GraphState:
        logger.info(f"[InferenceNode] Processing: {state['text'][:50]}...")
        try:
            predicted_label, confidence, full_results = self.classifier.predict(state['text'])
            logger.info(f"[InferenceNode] Predicted: {predicted_label} | Confidence: {confidence:.1%}")
            state.update({
                'predicted_label': predicted_label,
                'confidence': confidence,
                'full_results': full_results,
                'method_used': 'fine_tuned_model',
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"[InferenceNode] Error: {e}")
            state.update({
                'needs_fallback': True,
                'method_used': 'error_fallback'
            })
        return state


# ---- Confidence Check Node ----
class ConfidenceCheckNode:
    def __call__(self, state: GraphState) -> GraphState:
        confidence = state.get('confidence')
        if confidence is None:
            confidence = 0.0
        threshold = state.get('confidence_threshold', 0.7)
        if confidence < threshold:
            logger.info(f"[ConfidenceCheckNode] Confidence {confidence:.1%} < {threshold:.1%} → Trigger fallback")
            state['needs_fallback'] = True
        else:
            logger.info(f"[ConfidenceCheckNode] Confidence OK ({confidence:.1%} ≥ {threshold:.1%})")
            state['needs_fallback'] = False
            state['final_label'] = state['predicted_label']
        return state


# ---- Fallback Node ----
class FallbackNode:
    def __init__(self, model_path: str = ""):
        self.backup_classifier = BackupClassifier()
        self.primary_classifier = None
        if model_path:
            try:
                self.primary_classifier = SentimentClassifier(model_path)
            except Exception as e:
                logger.warning(f"Could not load primary classifier: {e}")

    def __call__(self, state: GraphState) -> GraphState:
        logger.info("[FallbackNode] Fallback activated")
        state['fallback_activated'] = True

        user_input = self._ask_user_clarification(state)
        if user_input and user_input.lower() not in ['skip', 'backup']:
            final_label = self._process_user_feedback(user_input, state)
            state.update({
                'user_feedback': user_input,
                'final_label': final_label,
                'method_used': 'user_clarification'
            })
            logger.info(f"[FallbackNode] Final label (user): {final_label}")
        else:
            logger.info("[FallbackNode] Using backup classifier")
            label, conf, results = self.backup_classifier.predict(state['text'])
            state.update({
                'final_label': label,
                'confidence': conf,
                'full_results': results,
                'method_used': 'backup_classifier'
            })
            logger.info(f"[FallbackNode] Backup prediction: {label} | Confidence: {conf:.1%}")
        return state

    def _ask_user_clarification(self, state: GraphState) -> str:
        print("\n" + "=" * 60)
        print("🤔 CLARIFICATION NEEDED")
        print("=" * 60)
        print(f"Text: {state['text']}")
        print(f"Initial prediction: {state.get('predicted_label', 'Unknown')} (Confidence: {state.get('confidence', 0.0):.1%})")
        print("\nThe model is unsure. Please help clarify the sentiment:")
        print("Options:")
        print("  - Type 'positive' for positive sentiment")
        print("  - Type 'negative' for negative sentiment")
        print("  - Type 'backup' to use the backup model")
        print("  - Type 'skip' to skip clarification")
        print("=" * 60)
        try:
            return input("Your input: ").strip()
        except (EOFError, KeyboardInterrupt):
            return "skip"

    def _process_user_feedback(self, user_input: str, state: GraphState) -> str:
        user_input_lower = user_input.lower()
        if any(word in user_input_lower for word in ['positive', 'good', 'yes']):
            return "POSITIVE"
        elif any(word in user_input_lower for word in ['negative', 'bad', 'no']):
            return "NEGATIVE"
        elif "was" in user_input_lower:
            if any(word in user_input_lower for word in ['not', 'bad', 'negative']):
                return "NEGATIVE"
            elif any(word in user_input_lower for word in ['good', 'positive']):
                return "POSITIVE"
        # Always return a string, never None
        return str(state.get('predicted_label')) if state.get('predicted_label') is not None else 'UNKNOWN'


# ---- Logging Node ----
class LoggingNode:
    def __init__(self, log_file: str = "classification_log.json"):
        self.log_file = log_file

    def __call__(self, state: GraphState) -> GraphState:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input_text": state['text'],
            "initial_prediction": state.get('predicted_label'),
            "initial_confidence": state.get('confidence'),
            "confidence_threshold": state.get('confidence_threshold'),
            "fallback_activated": state.get('fallback_activated', False),
            "user_feedback": state.get('user_feedback'),
            "final_label": state.get('final_label'),
            "method_used": state.get('method_used'),
            "full_results": state.get('full_results', {})
        }

        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"[LoggingNode] Error writing to log file: {e}")

        # Print result
        print(f"\n📊 FINAL RESULT")
        print("=" * 40)
        print(f"Input: {state['text']}")
        print(f"Final Label: {state.get('final_label', 'Unknown')}")
        print(f"Method: {state.get('method_used', 'Unknown')}")
        if state.get('confidence'):
            print(f"Confidence: {state['confidence']:.1%}")
        if state.get('fallback_activated'):
            print("Fallback: Activated")
        print("=" * 40 + "\n")

        return state


# ---- Routing Functions ----
def route_after_confidence_check(state: GraphState) -> str:
    return "fallback" if state.get('needs_fallback', False) else "logging"


def should_continue(state: GraphState) -> str:
    if state.get('final_label') is not None:
        return "logging"
    else:
        return "fallback"
