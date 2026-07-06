"""
Feedback endpoint — lets a reviewer (radiologist, tester, etc.) mark
whether a scan's prediction was correct, and optionally supply the
correct diagnosis. Feeds the roadmap's "user feedback / correction
mechanism" item and gives a foundation for future retraining data.

Save this as: app/routers/feedback.py
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app import db

router = APIRouter()


class FeedbackRequest(BaseModel):
    scan_db_id: int
    is_correct: bool
    corrected_diagnosis: Optional[str] = None
    comments: Optional[str] = None


@router.post('/feedback')
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit feedback on a previous scan's prediction.

    scan_db_id comes from the `scan_db_id` field returned by /api/predict.
    """
    try:
        db.save_feedback(
            scan_id=feedback.scan_db_id,
            is_correct=feedback.is_correct,
            corrected_diagnosis=feedback.corrected_diagnosis,
            comments=feedback.comments,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")

    return {"status": "ok", "message": "Feedback recorded, thank you."}
