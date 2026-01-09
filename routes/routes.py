import logging
import re
from typing import Dict, Any
from bson import ObjectId
from fastapi import APIRouter, UploadFile, File, HTTPException
from enums.Status import Status
from utils.pipeline import evaluation_pipeline
from io import BytesIO
import asyncio
from models.Task import Task
from db.database import db
from PyPDF2 import PdfReader
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
async def upload_file(cv: UploadFile = File(...), project_report: UploadFile = File(...)):
    try:
        cv_text = get_pdf_text(cv)
        report_text = get_pdf_text(project_report)
        task_data = Task(
            status=Status.QUEUED,
            cv=cv_text,
            report=report_text
        )

        new_task = await db["tasks"].insert_one(task_data.model_dump())

        return {
            "_id": str(new_task.inserted_id),
            "status": task_data.status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate/{task_id}")
async def evaluate(task_id: str):
    try:
        task = await db["tasks"].find_one(
            {"_id": ObjectId(task_id)}
        )
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if task["is_evaluate_running"] and task["status"] != Status.FAILED:
            if task["status"] == "completed":
                result = {"_id": str(task["_id"]), "status": task["status"]}
                cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", task["result"]["score"].strip(), flags=re.DOTALL)
                project_eval_cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", task["result"]["project_eval"].strip(), flags=re.DOTALL)
                result["result"]= json.loads(cleaned)
                result["result"]["cv_match_rate"] = result["result"]["cv_match_rate"]/100
                result["result"].update(json.loads(project_eval_cleaned))
                result["result"]["project_score"] = result["result"]["project_score"] * 2
                result["result"]["overall_summary"] = task["result"]["overall_summary"]
                return result
            return {"_id": str(task["_id"]), "status": task["status"]}

        asyncio.create_task(run_pipeline(task))
        await db["tasks"].update_one(
            {"_id": task["_id"]},
            {"$set": {"is_evaluate_running": True, "status": Status.QUEUED}}
        )
        return {"_id": str(task["_id"]), "status": Status.QUEUED}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def run_pipeline(task: Dict[str, Any]):
    try:
        await db["tasks"].update_one(
            {"_id": task["_id"]},
            {"$set": {"status": "processing"}},
        )
        result = await evaluation_pipeline(task["cv"], task["report"])
        await db["tasks"].update_one(
            {"_id": task["_id"]},
            {"$set": {"status": "completed", "result": result}}
        )
    except Exception as e:
        await db["tasks"].update_one(
            {"_id": task["_id"]},
            {"$set": {"status": "failed"}}
        )
        logging.error(f"Pipeline error for task {task['_id']}: {e}")


@router.get("/result/{task_id}")
async def get_result(task_id: str):
    try:
        task = await db["tasks"].find_one({"_id": ObjectId(task_id)})
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        result = {
            "_id": str(task["_id"]),
            "status": task["status"]
        }
        if task["result"] is not None:
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", task["result"]["score"].strip(), flags=re.DOTALL)
            project_eval_cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", task["result"]["project_eval"].strip(),
                                          flags=re.DOTALL)
            result["result"] = json.loads(cleaned)
            result["result"]["cv_match_rate"] = result["result"]["cv_match_rate"] / 100
            result["result"].update(json.loads(project_eval_cleaned))
            result["result"]["project_score"] = result["result"]["project_score"] * 2
            result["result"]["overall_summary"] = task["result"]["overall_summary"]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
