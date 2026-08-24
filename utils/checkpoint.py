import torch
from pathlib import Path
from typing import Dict, Any


class CheckpointManager:

    @staticmethod
    def save_checkpoint(models: Dict[str, Any], optimizers: Dict[str, Any],
                       metrics: Dict[str, Any], checkpoint_path: str,
                       epoch: int = None):
        checkpoint = {
            'epoch': epoch,
            'metrics': metrics,
        }

        for model_name, model in models.items():
            if hasattr(model, 'state_dict'):
                checkpoint[f'{model_name}_state'] = model.state_dict()

        for opt_name, optimizer in optimizers.items():
            if hasattr(optimizer, 'state_dict'):
                checkpoint[f'{opt_name}_state'] = optimizer.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    @staticmethod
    def load_checkpoint(checkpoint_path: str, models: Dict[str, Any],
                       optimizers: Dict[str, Any] = None):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        for model_name, model in models.items():
            if hasattr(model, 'load_state_dict') and f'{model_name}_state' in checkpoint:
                model.load_state_dict(checkpoint[f'{model_name}_state'])

        if optimizers is not None:
            for opt_name, optimizer in optimizers.items():
                if hasattr(optimizer, 'load_state_dict') and f'{opt_name}_state' in checkpoint:
                    optimizer.load_state_dict(checkpoint[f'{opt_name}_state'])

        metrics = checkpoint.get('metrics', {})
        epoch = checkpoint.get('epoch', None)

        print(f"Checkpoint loaded from {checkpoint_path}")
        return epoch, metrics

    @staticmethod
    def save_best_model(model, filepath: str, metric_value: float,
                       best_metric: float = None):
        if best_metric is None or metric_value < best_metric:
            torch.save({
                'model': model.state_dict(),
                'metric': metric_value,
            }, filepath)
            print(f"Best model saved with metric={metric_value:.4f}")
            return metric_value

        return best_metric

    @staticmethod
    def load_best_model(model, filepath: str):
        checkpoint = torch.load(filepath, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        metric = checkpoint.get('metric', None)
        print(f"Best model loaded with metric={metric}")
        return model, metric