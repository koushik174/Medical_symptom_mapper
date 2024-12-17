import os
import torch
import json
from datetime import datetime

class ModelManager:
    def __init__(self, base_path='models/saved_models'):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def save_model(self, model, tokenizer, condition_labels, metrics=None):
        """Save model, tokenizer, labels and metrics"""
        # Create timestamp for version control
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = os.path.join(self.base_path, f'model_{timestamp}')
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model state
        torch.save(model.state_dict(), os.path.join(model_dir, 'model_state.pt'))
        
        # Save tokenizer
        tokenizer.save_pretrained(os.path.join(model_dir, 'tokenizer'))
        
        # Save condition labels
        with open(os.path.join(model_dir, 'condition_labels.json'), 'w') as f:
            json.dump(condition_labels.tolist(), f)
        
        # Save metrics if provided
        if metrics:
            with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f)
        
        print(f"Model saved in {model_dir}")
        return model_dir
    
    def load_model(self, model_dir):
        """Load model, tokenizer and labels"""
        # Load model state
        model_state = torch.load(os.path.join(model_dir, 'model_state.pt'))
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, 'tokenizer'))
        
        # Load condition labels
        with open(os.path.join(model_dir, 'condition_labels.json'), 'r') as f:
            condition_labels = json.load(f)
        
        return model_state, tokenizer, condition_labels

# Example usage after training:
manager = ModelManager()

# Save the model
final_metrics = {
    'train_loss': train_history[-1]['loss'],
    'val_loss': val_history[-1]['loss'],
    'val_f1': val_history[-1]['metrics']['f1']
}

saved_dir = manager.save_model(
    model=model,
    tokenizer=tokenizer,
    condition_labels=condition_labels,
    metrics=final_metrics
)
