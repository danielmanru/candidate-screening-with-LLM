from typing import Optional, Dict, Any

from pydantic import BaseModel

from enums.Status import Status


class Task(BaseModel):
    status: Status
    cv: str
    is_evaluate_running: Optional[bool] = False
    report: Optional[str] = None
    result: Optional[Dict[str, Any]] = None