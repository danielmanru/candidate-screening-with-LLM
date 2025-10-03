from typing import Optional, Dict, Any

from pydantic import BaseModel

class Task(BaseModel):
    status: str
    cv: str
    report: Optional[str] = None
    result: Optional[Dict[str, Any]] = None