# CurrentData implementation
# File name current_data.py
from dataclasses import dataclass

@dataclass
class CurrentData:
    """Base class for holding current market data and indicators for a strategy step."""
    price: float
    time: str
    # Add other common data fields if necessary