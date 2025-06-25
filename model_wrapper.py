"""
Model wrapper for inference with confidence scoring
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

class SentimentClassifier:
    def __init__(self, model_path: str, base_model_name: str = "distilbert-base-uncased"):
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
        
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load base model
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.base_model_name,
                num_labels=2,
                id2label={0: "NEGATIVE", 1: "POSITIVE"},
                label2id={"NEGATIVE": 0, "POSITIVE": 1}
            )
            
            # Load LoRA weights
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def predict(self, text: str) -> Tuple[str, float, Dict]:
        """
        Predict sentiment with confidence score
        
        Returns:
            Tuple of (predicted_label, confidence_score, full_results)
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded")
            
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Calculate probabilities and confidence
            probabilities = F.softmax(logits, dim=-1)
            confidence, predicted_class = torch.max(probabilities, dim=-1)
            
            # Convert to human-readable format
            predicted_label = self.model.config.id2label[predicted_class.item()]
            confidence_score = confidence.item()
            
            # Detailed results
            results = {
                "text": text,
                "predicted_label": predicted_label,
                "confidence": confidence_score,
                "probabilities": {
                    "NEGATIVE": probabilities[0][0].item(),
                    "POSITIVE": probabilities[0][1].item()
                },
                "raw_logits": logits[0].tolist()
            }
            
        return predicted_label, confidence_score, results
        
    def batch_predict(self, texts: list) -> list:
        """Predict sentiment for multiple texts"""
        results = []
        for text in texts:
            pred_label, confidence, full_results = self.predict(text)
            results.append(full_results)
        return results

class BackupClassifier:
    """Simple rule-based backup classifier"""
    
    def __init__(self):
        self.positive_words = set([
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied', 'perfect',
            'awesome', 'brilliant', 'outstanding', 'superb', 'magnificent'
        ])
        
        self.negative_words = set([
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'boring',
            'slow', 'painful', 'disappointing', 'worse', 'worst', 'annoying',
            'frustrating', 'useless', 'pathetic', 'ridiculous', 'stupid'
        ])
        
    def predict(self, text: str) -> Tuple[str, float, Dict]:
        """Simple rule-based prediction"""
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        if positive_count > negative_count:
            label = "POSITIVE"
            confidence = min(0.8, 0.5 + (positive_count - negative_count) * 0.1)
        elif negative_count > positive_count:
            label = "NEGATIVE"  
            confidence = min(0.8, 0.5 + (negative_count - positive_count) * 0.1)
        else:
            label = "NEUTRAL"  # Default to neutral
            confidence = 0.5
            
        results = {
            "text": text,
            "predicted_label": label,
            "confidence": confidence,
            "method": "rule_based_backup",
            "positive_words_found": positive_count,
            "negative_words_found": negative_count
        }
        
        return label, confidence, results