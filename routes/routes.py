import re
from typing import Dict, Any

from bson import ObjectId
from fastapi import APIRouter, UploadFile, File, HTTPException
from services.pipeline import evaluation_pipeline
from io import BytesIO
import asyncio
from models.tasks import Task
from db.database import db
from PyPDF2 import PdfReader
from services.pipeline import run_evaluation
import json

router = APIRouter()


def task_helper(task) -> dict:
    result: Dict[str, Any] = {}

    for key, value in task.items():
        if key == "_id":
            result[key] = str(value)
        else:
            result[key] = value
    return result

def get_pdf_text(pdf_docs: UploadFile):
    text = ""
    pdf_reader = PdfReader(BytesIO(pdf_docs.file.read()))
    pdf_docs.file.seek(0)
    for page in pdf_reader.pages:
        page_text = page.extract_text() or ""
        text += page_text
    return text


@router.post("/upload")
# async def upload_file(cv: UploadFile = File(...), project_report: UploadFile = File(...)):
async def upload_file(cv: UploadFile = File(...)):
    cv_text = get_pdf_text(cv)
    # report_text = get_pdf_text(project_report)
    task_data = Task(
        status="queued",
        cv=cv_text,
        # report=report_text
    )

    new_task = await db["tasks"].insert_one(task_data.model_dump())
    created = await db["tasks"].find_one(
        {"_id": new_task.inserted_id},
        projection={
            "_id": 1,
            "status": 1,
        }
    )
    asyncio.create_task(run_evaluation(str(created["_id"])))
    return {"_id": str(created["_id"]), "status": created["status"]}


@router.post("/evaluate/{task_id}")
async def evaluate(task_id: str):
    result = await run_evaluation(task_id)
    return result

@router.get("/result/{task_id}")
async def get_result(task_id: str):
    task = await db["tasks"].find_one({"_id": ObjectId(task_id)})
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task_converted = task_helper(task)
    if task_converted["result"] is not None:
        cleaned = re.sub(r"^```json\s*|\s*```$", "", task_converted["result"]["score"].strip(), flags=re.DOTALL)
        task_converted["result"]["score"] = json.loads(cleaned)
    return task_converted
