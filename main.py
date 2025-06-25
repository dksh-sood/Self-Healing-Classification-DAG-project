"""
Main CLI interface for self-healing classification system
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from colorama import init, Fore, Style, Back
from dag_workflow import create_dag
import json
from datetime import datetime

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('classification_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ClassificationCLI:
    """Command-line interface for the classification system"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.7):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.dag = None
        self._initialize_dag()
        
    def _initialize_dag(self):
        """Initialize the DAG"""
        try:
            print(f"{Fore.CYAN}🤖 Initializing Self-Healing Classification System...")
            self.dag = create_dag(self.model_path, self.confidence_threshold)
            print(f"{Fore.GREEN}✅ System initialized successfully!")
        except Exception as e:
            print(f"{Fore.RED}❌ Failed to initialize system: {e}")
            sys.exit(1)
            
    def display_welcome(self):
        """Display welcome message"""
        print(f"\n{Back.BLUE}{Fore.WHITE}" + "="*70)
        print(f"{Back.BLUE}{Fore.WHITE}🎯 SELF-HEALING SENTIMENT CLASSIFICATION SYSTEM")
        print(f"{Back.BLUE}{Fore.WHITE}   Built with LangGraph & Fine-tuned Transformer")
        print(f"{Back.BLUE}{Fore.WHITE}" + "="*70 + Style.RESET_ALL)
        print(f"\n{Fore.YELLOW}Features:")
        print(f"  🧠 Fine-tuned DistilBERT with LoRA")
        print(f"  🔄 Self-healing with confidence-based fallback")
        print(f"  👤 Human-in-the-loop clarification")
        print(f"  🔧 Backup rule-based classifier")
        print(f"  📊 Structured logging")
        print(f"\n{Fore.CYAN}Confidence Threshold: {self.confidence_threshold:.1%}")
        
    def display_help(self):
        """Display help information"""
        print(f"\n{Fore.CYAN}📖 AVAILABLE COMMANDS:")
        print(f"  {Fore.GREEN}classify <text>{Fore.WHITE}  - Classify sentiment of text")
        print(f"  {Fore.GREEN}batch{Fore.WHITE}            - Enter batch processing mode")
        print(f"  {Fore.GREEN}stats{Fore.WHITE}            - Show classification statistics")
        print(f"  {Fore.GREEN}workflow{Fore.WHITE}         - Display DAG workflow structure")
        print(f"  {Fore.GREEN}help{Fore.WHITE}             - Show this help message")
        print(f"  {Fore.GREEN}quit{Fore.WHITE}             - Exit the system")
        print(f"\n{Fore.YELLOW}Example: {Fore.WHITE}classify The movie was absolutely amazing!")
        
    def classify_text(self, text: str):
        """Classify a single text input"""
        if not text.strip():
            print(f"{Fore.RED}❌ Please provide text to classify")
            return
            
        print(f"\n{Fore.CYAN}🔍 Processing: {Fore.WHITE}{text}")
        print(f"{Fore.CYAN}{'='*60}")
        
        try:
            # Run the DAG
            result = self.dag.run(text)
            
            # The logging is handled within the DAG nodes
            return result
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error during classification: {e}")
            logger.error(f"Classification error: {e}")
            
    def batch_process(self):
        """Enter batch processing mode"""
        print(f"\n{Fore.CYAN}📝 BATCH PROCESSING MODE")
        print(f"{Fore.YELLOW}Enter texts one by one. Type 'done' when finished.")
        print(f"{'='*50}")
        
        batch_results = []
        while True:
            try:
                text = input(f"{Fore.GREEN}Enter text (or 'done'): {Fore.WHITE}").strip()
                if text.lower() == 'done':
                    break
                if text:
                    result = self.classify_text(text)
                    if result:
                        batch_results.append(result)
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Batch processing interrupted")
                break
                
        if batch_results:
            self.display_batch_summary(batch_results)
            
    def display_batch_summary(self, results: list):
        """Display summary of batch processing"""
        print(f"\n{Fore.CYAN}📊 BATCH PROCESSING SUMMARY")
        print(f"{'='*50}")
        
        total = len(results)
        fallback_count = sum(1 for r in results if r.get('fallback_activated', False))
        positive_count = sum(1 for r in results if r.get('final_label') == 'POSITIVE')
        negative_count = sum(1 for r in results if r.get('final_label') == 'NEGATIVE')
        
        print(f"Total processed: {total}")
        print(f"Fallback activated: {fallback_count} ({fallback_count/total*100:.1f}%)")
        print(f"Positive sentiment: {positive_count} ({positive_count/total*100:.1f}%)")
        print(f"Negative sentiment: {negative_count} ({negative_count/total*100:.1f}%)")
        
    def show_stats(self):
        """Show classification statistics from log file"""
        log_file = "classification_log.json"
        
        if not os.path.exists(log_file):
            print(f"{Fore.YELLOW}📊 No statistics available yet. Run some classifications first!")
            return
            
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = [json.loads(line) for line in f if line.strip()]
                
            if not logs:
                print(f"{Fore.YELLOW}📊 No statistics available yet.")
                return
                
            print(f"\n{Fore.CYAN}📊 CLASSIFICATION STATISTICS")
            print(f"{'='*50}")
            
            total = len(logs)
            fallback_count = sum(1 for log in logs if log.get('fallback_activated', False))
            
            # Method distribution
            methods = {}
            for log in logs:
                method = log.get('method_used', 'unknown')
                methods[method] = methods.get(method, 0) + 1
                
            # Label distribution
            labels = {}
            for log in logs:
                label = log.get('final_label', 'unknown')
                labels[label] = labels.get(label, 0) + 1
                
            # Confidence statistics
            confidences = [log.get('initial_confidence', 0) for log in logs if log.get('initial_confidence')]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            print(f"Total Classifications: {total}")
            print(f"Fallback Rate: {fallback_count/total*100:.1f}% ({fallback_count}/{total})")
            print(f"Average Confidence: {avg_confidence:.1%}")
            print(f"\nMethod Distribution:")
            for method, count in methods.items():
                print(f"  {method}: {count} ({count/total*100:.1f}%)")
            print(f"\nLabel Distribution:")
            for label, count in labels.items():
                print(f"  {label}: {count} ({count/total*100:.1f}%)")
                
        except Exception as e:
            print(f"{Fore.RED}❌ Error reading statistics: {e}")
            
    def run_interactive(self):
        """Run the interactive CLI"""
        self.display_welcome()
        self.display_help()
        
        while True:
            try:
                user_input = input(f"\n{Fore.GREEN}> {Fore.WHITE}").strip()
                
                if not user_input:
                    continue
                    
                # Parse command
                parts = user_input.split(' ', 1)
                command = parts[0].lower()
                
                if command == 'quit' or command == 'exit':
                    print(f"{Fore.CYAN}👋 Goodbye!")
                    break
                    
                elif command == 'help':
                    self.display_help()
                    
                elif command == 'classify':
                    if len(parts) > 1:
                        self.classify_text(parts[1])
                    else:
                        print(f"{Fore.RED}Usage: classify <text>")
                        
                elif command == 'batch':
                    self.batch_process()
                    
                elif command == 'stats':
                    self.show_stats()
                    
                elif command == 'workflow':
                    self.dag.visualize_workflow()
                    
                else:
                    # Treat unknown command as text to classify
                    self.classify_text(user_input)
                    
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Use 'quit' to exit")
            except EOFError:
                print(f"\n{Fore.CYAN}👋 Goodbye!")
                break

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Self-Healing Sentiment Classification System")
    parser.add_argument(
        "--model-path", 
        default="./fine_tuned_model",
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for fallback activation"
    )
    parser.add_argument(
        "--text",
        help="Text to classify (non-interactive mode)"
    )
    parser.add_argument(
        "--batch-file",
        help="File containing texts to classify (one per line)"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"{Fore.RED}❌ Model not found at {args.model_path}")
        print(f"{Fore.YELLOW}💡 Run 'python finetune_model.py' first to train the model")
        sys.exit(1)
    
    # Initialize CLI
    cli = ClassificationCLI(args.model_path, args.confidence_threshold)
    
    # Handle different modes
    if args.text:
        # Single text classification
        cli.classify_text(args.text)
    elif args.batch_file:
        # Batch file processing
        if os.path.exists(args.batch_file):
            with open(args.batch_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            print(f"{Fore.CYAN}Processing {len(texts)} texts from {args.batch_file}")
            results = []
            for text in texts:
                result = cli.classify_text(text)
                if result:
                    results.append(result)
            
            cli.display_batch_summary(results)
        else:
            print(f"{Fore.RED}❌ Batch file not found: {args.batch_file}")
    else:
        # Interactive mode
        cli.run_interactive()

if __name__ == "__main__":
    main()