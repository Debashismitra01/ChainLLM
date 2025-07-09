from enum import Enum
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any, Optional, Dict
from db import db
from executor import execute_workflow


app = FastAPI()


# Enum for task status
class Status(str, Enum):
    pending = "pending"
    completed = "completed"
    failed = "failed"

# Pydantic model for valid fields
class Data(BaseModel):
    task_id: str
    workflow: dict[str, Any]
    result: dict[str, Any] = {}
    status: Status = Field(default=Status.pending)
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    class Config:
        extra = "ignore"  # Ignore any other fields


# Function to clean raw JSON input
def clean_data(raw: dict) -> tuple[Data, str]:
    session_token = raw.get('session_token')  # Just to read, not store
    data_dict = {
        "task_id": raw.get("task_id"),
        "workflow": raw.get("workflow"),
    }

    validated_data = Data(**data_dict)
    return validated_data, session_token


@app.get("/")
async def home():
    return {"message": "hello world"}

# ✅ This should be POST
@app.post("/api")
async def api(request: Request):
    try:
        raw = await request.json()
        data, session_token = clean_data(raw)

        check = await db["tasks"].find_one({"task_id": data.task_id})
        if check:
            raise HTTPException(status_code=409, detail="duplicate task id")

        env, results, success, completed_at  = await execute_workflow(data.workflow)

        # ✅ Assign results back to the Pydantic model
        data.result = {
            "env": env,
            "steps": results,
            "success": success
        }
        data.status = Status.completed if success else Status.failed
        data.completed_at = completed_at

        # ✅ Now save the updated data
        await db["tasks"].insert_one(data.dict())

        return {
            "message": "Data Stored",
            "task_id": data.task_id,
            "session_token": session_token
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error storing data: {e}")




@app.get("/api/{task_id}")
async def retrive(task_id: str):
    result = await db["tasks"].find_one({"task_id": task_id})
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
    result["_id"] = str(result["_id"])  # Make Mongo _id JSON serializable
    return result
