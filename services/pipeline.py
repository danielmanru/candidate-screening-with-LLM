import asyncio
from typing import Dict, Any, Optional
from bson import ObjectId
from fastapi import HTTPException
from db.database import db
from db.vector import retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

async def run_evaluation(task_id: str):
    task = await db["tasks"].find_one(
        {"_id": ObjectId(task_id)}
    )
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] == "processing":
        raise HTTPException(status_code=409, detail="Task is being processing")
    if task["status"] == "completed":
        raise HTTPException(status_code=409, detail="The task is complete")

    async def run_pipeline():
        try:
            # result = await evaluation_pipeline(task["cv"], task["report"])
            result = await evaluation_pipeline(task["cv"])
            await db["tasks"].update_one(
                {"_id": task["_id"]},
                {"$set": {"status": "completed", "result": result}}
            )
        except Exception as e:
            await db["tasks"].update_one(
                {"_id": task["_id"]},
                {"$set": {"status": "failed"}}
            )
            print(f"Pipeline error for task {task['_id']}: {e}")

    asyncio.create_task(run_pipeline())
    await db["tasks"].update_one(
        {"_id": task["_id"]},
        {"$set": {"status": "processing"}},
    )

    return {"_id": str(task["_id"]), "status": "processing"}

async def llm_call(template: str, data: Optional[Dict[str, Any]] = None) -> str:
    raw_key = os.getenv("OPENROUTER_API_KEY")
    model = ChatOpenAI(
        model="deepseek/deepseek-chat-v3.1:free",
        api_key=raw_key,
        base_url=os.getenv("OPENROUTER_BASE_URL"),
        temperature=0.5
    )
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    result = await chain.ainvoke(data)
    return result.content


# async def evaluation_pipeline(cv_text: str, report_text: str):
async def evaluation_pipeline(cv_text: str):
    # Step 1: Extract
    extracted_template = """Extract skills, experiences, projects from CV: {cv_text}"""
    extracted = await llm_call(extracted_template, {"cv_text": cv_text})

    # Step 2: Compare
    job_docs = await retriever.ainvoke("job description")
    comparison_template = """Compare extracted CV: {extracted} with job desc: {job_docs}, determine whether the candidate meets the qualifications for the job."""
    comparison = await llm_call(comparison_template, {"extracted": extracted, "job_docs": job_docs[0].page_content})

    # Step 3: Score & feedback
    score_rubric = await retriever.ainvoke("score rubric")
    score_feedback_template = """
    Based on the comparison between the CV and the job description below: {comparison},
    evaluate the candidate's resume by following this scoring rubric: {score_rubric}.
    Then provide feedback in 3-5 sentences (strengths, weaknesses, recommendations). 
    Provide the results in json format containing cv_match_rate and cv_feedback.
    """
    score_feedback = await llm_call(score_feedback_template, {"comparison": comparison, "score_rubric": score_rubric[0].page_content})

    # Step 4: Project report eval
    # project_eval = fake_llm_call(
    #     f"Evaluate report with rubric: {rubric_docs[0].page_content}\nReport: {report_text}"
    # )
    #
    # refined = fake_llm_call(f"Refine evaluation: {project_eval}")

    return {
        "extracted": extracted,
        "comparison": comparison,
        "score": score_feedback,
        # "feedback": feedback,
        # "project_eval": refined,
    }