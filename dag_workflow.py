"""
LangGraph workflow for self-healing classification
"""
from langgraph.graph import StateGraph, END
from dag_nodes import (
    GraphState, InferenceNode, ConfidenceCheckNode, 
    FallbackNode, LoggingNode, route_after_confidence_check
)
import logging

logger = logging.getLogger(__name__)

class SelfHealingClassificationDAG:
    """Main DAG class for self-healing classification"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.7):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.workflow = None
        self.app = None
        self._build_workflow()
        
    def _build_workflow(self):
        """Build the LangGraph workflow"""
        logger.info("Building classification DAG...")
        
        # Initialize nodes
        inference_node = InferenceNode(self.model_path)
        confidence_check_node = ConfidenceCheckNode()
        fallback_node = FallbackNode(self.model_path)
        logging_node = LoggingNode()
        
        # Create the graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("inference", inference_node)
        workflow.add_node("confidence_check", confidence_check_node)
        workflow.add_node("fallback", fallback_node)
        workflow.add_node("logging", logging_node)
        
        # Set entry point
        workflow.set_entry_point("inference")
        
        # Add edges
        workflow.add_edge("inference", "confidence_check")
        
        # Conditional edge after confidence check
        workflow.add_conditional_edges(
            "confidence_check",
            route_after_confidence_check,
            {
                "fallback": "fallback",
                "logging": "logging"
            }
        )
        
        # Both fallback and logging end the workflow
        workflow.add_edge("fallback", "logging")
        workflow.add_edge("logging", END)
        
        # Compile the workflow
        self.app = workflow.compile()
        self.workflow = workflow
        
        logger.info("DAG built successfully")
        
    def run(self, text: str) -> dict:
        """Run the classification workflow"""
        initial_state = GraphState(
            text=text,
            predicted_label=None,
            confidence=None,
            full_results=None,
            needs_fallback=False,
            fallback_activated=False,
            user_feedback=None,
            final_label=None,
            method_used="",
            timestamp="",
            confidence_threshold=self.confidence_threshold
        )
        
        # Execute the workflow
        final_state = self.app.invoke(initial_state)
        
        return final_state
        
    def visualize_workflow(self):
        """Print workflow structure"""
        print("\n🔄 CLASSIFICATION DAG STRUCTURE")
        print("="*50)
        print("1. InferenceNode → Run fine-tuned model")
        print("2. ConfidenceCheckNode → Evaluate confidence")
        print("3. [Conditional] → High confidence: Skip to logging")
        print("4. [Conditional] → Low confidence: Go to fallback")
        print("5. FallbackNode → User clarification or backup model")
        print("6. LoggingNode → Log results and display")
        print("="*50)
        print(f"Confidence Threshold: {self.confidence_threshold:.1%}")
        print("="*50)

def create_dag(model_path: str, confidence_threshold: float = 0.7) -> SelfHealingClassificationDAG:
    """Factory function to create the DAG"""
    return SelfHealingClassificationDAG(model_path, confidence_threshold)