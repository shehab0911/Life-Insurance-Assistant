import os
import json
import time
import io
import base64
import asyncio
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from openai import OpenAI

# Internal imports
from .langgraph_agent import run_agent

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in .env file.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# Ensure directories exist
os.makedirs(os.path.join(BASE_DIR, "frontend", "templates"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "frontend", "static"), exist_ok=True)

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "frontend", "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "frontend", "static")), name="static")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("--- WebSocket Connected ---")
    
    # Default session ID for this connection
    session_id = "web_session_" + str(int(time.time()))

    try:
        while True:
            try:
                raw = await websocket.receive_text()
                msg = json.loads(raw)
            except WebSocketDisconnect:
                print("Client disconnected")
                break
            except Exception as e:
                print(f"Error receiving: {e}")
                break

            typ = msg.get("type")
            
            # Allow client to override session_id if they want, otherwise use connection default
            client_session = msg.get("session_id")
            if client_session:
                session_id = client_session

            # --- HANDLE RESET ---
            if typ == "reset":
                await websocket.send_text(json.dumps({"type":"system", "text":"Session reset."}))
                continue

            # --- HANDLE AUDIO ---
            if typ == "audio":
                print("\n--- ⚡ PROCESSING AUDIO ---")
                start_time = time.time()
                b64 = msg.get("data")
                
                try:
                    if "," in b64: 
                        b64 = b64.split(",")[1]
                    audio_data = base64.b64decode(b64)
                    audio_file = io.BytesIO(audio_data)
                    audio_file.name = "audio.webm" 
                    
                    # 1. Transcribe
                    def transcribe_sync():
                        return openai_client.audio.transcriptions.create(
                            model="whisper-1", file=audio_file, language="en" 
                        )
                    transcript_obj = await asyncio.to_thread(transcribe_sync)
                    user_text = transcript_obj.text.strip()
                    
                    # Send transcript back to UI immediately
                    await websocket.send_text(json.dumps({"type":"transcript", "text": user_text}))

                    # 2. AI Response
                    answer = await run_agent(user_text, session_id)
                    
                    # Send response
                    await websocket.send_text(json.dumps({"type":"response", "text": answer}))
                    print(f"Total Time: {(time.time()-start_time):.2f}s")

                except Exception as e:
                    print(f"Audio Error: {e}")
                    await websocket.send_text(json.dumps({"type":"error", "message":str(e)}))

            # --- HANDLE TEXT (Chat CLI via UI) ---
            elif typ == "text":
                print("\n--- 💬 PROCESSING TEXT ---")
                user_text = msg.get("data", "").strip()
                if not user_text:
                    continue

                try:
                    # 1. AI Response
                    answer = await run_agent(user_text, session_id)
                    # 2. Send response
                    await websocket.send_text(json.dumps({"type":"response", "text": answer}))
                except Exception as e:
                    await websocket.send_text(json.dumps({"type":"error", "message":str(e)}))

    except Exception as e:
        print(f"WebSocket Error: {e}")