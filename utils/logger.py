import json
from datetime import datetime
from pathlib import Path


class Logger:

    def __init__(self, log_dir='./logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.logs = []

    def log(self, **kwargs):
        entry = {
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        self.logs.append(entry)

    def save(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)

    def get_logs(self):
        return self.logs