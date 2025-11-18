
import base64, os, uuid, tempfile
from typing import Tuple

def save_base64_wav(base64_data: str, dst_folder: str = None) -> str:
   
    header, data = base64_data.split(",", 1) if "," in base64_data else ("", base64_data)
    raw = base64.b64decode(data)
    folder = dst_folder or tempfile.gettempdir()
    filename = os.path.join(folder, f"{uuid.uuid4().hex}.webm")
    with open(filename, "wb") as f:
        f.write(raw)
    return filename
