"""
Fine-tuning script for sentiment classification using LoRA
"""
import os
import json
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentFineTuner:
    def __init__(self, model_name="distilbert-base-uncased", max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.dataset = None
        
    def load_dataset(self):
        """Load and prepare the IMDB sentiment dataset"""
        logger.info("Loading IMDB dataset...")
        dataset = load_dataset("imdb")
        
        # Take a smaller subset for faster training (optional)
        train_size = 5000
        test_size = 1000
        
        self.dataset = DatasetDict({
            'train': dataset['train'].shuffle(seed=42).select(range(train_size)),
            'test': dataset['test'].shuffle(seed=42).select(range(test_size))
        })
        
        logger.info(f"Dataset loaded: {len(self.dataset['train'])} train, {len(self.dataset['test'])} test samples")
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        logger.info(f"Loading model and tokenizer: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            id2label={0: "NEGATIVE", 1: "POSITIVE"},
            label2id={"NEGATIVE": 0, "POSITIVE": 1}
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def setup_lora(self):
        """Setup LoRA configuration"""
        logger.info("Setting up LoRA configuration...")
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin"]  # DistilBERT specific
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def tokenize_function(self, examples):
        """Tokenize the text"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized. Call setup_model_and_tokenizer() first.")
        if "text" not in examples:
            raise KeyError(f"Expected 'text' key in examples, got keys: {list(examples.keys())}")
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=self.max_length
        )
        
    def prepare_dataset(self):
        """Tokenize and prepare dataset"""
        logger.info("Tokenizing dataset...")
        if self.dataset is None:
            raise ValueError("Dataset is not loaded. Call load_dataset() first.")
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized. Call setup_model_and_tokenizer() first.")
        tokenized_dataset = self.dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        return tokenized_dataset
        
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        
    def train(self, output_dir="./fine_tuned_model"):
        """Train the model"""
        logger.info("Starting training...")
        
        tokenized_dataset = self.prepare_dataset()
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-4,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            warmup_steps=500,
            save_total_limit=2,
            report_to=None,  # Disable wandb
        )
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training results
        results = trainer.evaluate()
        with open(f"{output_dir}/training_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Training completed. Model saved to {output_dir}")
        logger.info(f"Final results: {results}")
        
        return trainer

def main():
    """Main training function"""
    fine_tuner = SentimentFineTuner()
    try:
        # Load dataset
        fine_tuner.load_dataset()
        # Setup model and tokenizer
        fine_tuner.setup_model_and_tokenizer()
        # Setup LoRA
        fine_tuner.setup_lora()
        # Train
        trainer = fine_tuner.train()
        logger.info("Fine-tuning completed successfully!")
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")

if __name__ == "__main__":
    main()