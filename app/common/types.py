from dataclasses import dataclass
from pandas import DataFrame
import base64
from io import StringIO
from typing import Dict, Any
import pandas as pd


@dataclass
class DocumentMetadata:

    def __init__(self, file_name: str, df: DataFrame):
        self.file_name = file_name
        self.df = df
        self.columns = df.columns
        self.num_rows = len(df)
        self.column_names = [str(col) for col in df.columns]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize DocumentMetadata to a dictionary format."""
        # Convert DataFrame to CSV string
        csv_buffer = StringIO()
        self.df.to_csv(csv_buffer, index=False)
        csv_str = csv_buffer.getvalue()
        
        # Encode as base64
        serialized_df = base64.b64encode(csv_str.encode()).decode()
        
        return {
            "file_name": self.file_name,
            "columns": [str(col) for col in self.columns],
            "num_rows": self.num_rows,
            "serialized_df": serialized_df
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentMetadata':
        """Create DocumentMetadata instance from a dictionary.
        
        Args:
            data: Dictionary containing serialized DocumentMetadata
            
        Returns:
            DocumentMetadata instance
            
        Raises:
            ValueError: If required fields are missing or data is invalid
        """
        if not all(key in data for key in ["file_name", "serialized_df"]):
            raise ValueError("Missing required fields in data dictionary")
            
        try:
            # Decode base64 CSV string back to DataFrame
            csv_str = base64.b64decode(data["serialized_df"]).decode()
            df = pd.read_csv(StringIO(csv_str))
            
            return cls(
                file_name=data["file_name"],
                df=df
            )
        except Exception as e:
            raise ValueError(f"Error deserializing DataFrame: {str(e)}")
