from typing import Dict, Any, Optional
from db.vector import retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

async def llm_call(template: str, data: Optional[Dict[str, Any]] = None) -> str:
    model = ChatOpenAI(
        model="meta-llama/llama-3.3-70b-instruct:free",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL"),
        temperature=0.2
    )
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    result = await chain.ainvoke(data)
    return result.content

async def evaluation_pipeline(cv_text: str, report_text: str) -> Dict[str, Any]:
    # Step 1: Extract
    extracted_template = """Extract skills, experiences, projects from CV: {cv_text}"""
    extracted = await llm_call(extracted_template, {"cv_text": cv_text})

    # Step 2: Compare
    job_docs = await retriever.ainvoke("About You")
    comparison_template = """Compare extracted CV: {extracted} with job desc: {job_docs}, determine whether the candidate meets the qualifications for the job."""
    comparison = await llm_call(comparison_template, {"extracted": extracted, "job_docs": job_docs[0].page_content})

    # Step 3: Score & feedback
    score_rubric = await retriever.ainvoke("CV Match Evaluation")
    score_feedback_template = """
    Evaluate the candidate's CV based on the comparison and rubric below
    Return the result **only** as valid JSON with the following structure:
    {{
        "cv_match_rate": <numerical_score>,
        "cv_feedback": "<3-5 sentence feedback including summarizing strengths, weaknesses, and recommendations in single paragraph>"
    }}
    Comparison:
    {comparison}

    Scoring rubric:
    {score_rubric}

    Do not include any explanation outside of the JSON object.
    """

    score_feedback = await llm_call(score_feedback_template, {"comparison": comparison, "score_rubric": score_rubric[0].page_content})

    # Step 4: Project report eval
    project_score_rubric = await retriever.ainvoke("Project Deliverable Evaluation")
    project_eval_template = """
    Evaluate this project report based on the rubric below.
    Return the result **only** as valid JSON with the following structure:
    {{
        "project_score": <numerical_score>,
        "project_feedback": "<3-5 sentence feedback including summarizing strengths, weaknesses, and recommendations in single paragraph>"
    }}
    Report text:
    {report_text}

    Scoring rubric:
    {project_score_rubric}

    Do not include any explanation outside of the JSON object.
    """
    project_eval = await llm_call(project_eval_template,{"report_text": report_text, "project_score_rubric": project_score_rubric[0].page_content}
    )

    # Step 5: Overall Summary
    overall_summary_template = """
    Based on the assessment and feedback for CV {score_feedback} 
    and for project report {project_eval}, 
    write an overall summary including strengths, weaknesses, and recommendations.
    
    Output requirements:
    - Exactly 3–5 sentences.
    - Single paragraph, no line breaks or lists.
    - Plain text only.
    
    Return only the paragraph.
    """
    overall_summary = await llm_call(overall_summary_template, {"score_feedback": score_feedback, "project_eval": project_eval})

    return {
        "extracted": extracted,
        "comparison": comparison,
        "score": score_feedback,
        "project_eval": project_eval,
        "overall_summary": overall_summary,
    }