from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class PingRequest(BaseModel):
    message: str

class PingResponse(BaseModel):
    message: str



@app.post("/ping", response_model=PingResponse)
def ping(req: PingRequest) -> PingResponse:

    return PingResponse(message=f"pong: {req.message}")
