from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import services.llm as llm

app = FastAPI()

class PingRequest(BaseModel):
    message: str

class PingResponse(BaseModel):
    message: str

class AskResponse(BaseModel):
    message: str
    tokens: int

class AskRequest(BaseModel):
    text:str


@app.post("/ping", response_model=PingResponse)
def ping(req: PingRequest) -> PingResponse:

    return PingResponse(message=f"pong: {req.message}")


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:

    response, tokens = await llm.claude_complete(prompt= req.text)
    return AskResponse(message=f"{response}", tokens=tokens)



