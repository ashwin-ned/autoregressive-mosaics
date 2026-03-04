import re
import os
import math
import time
import logging
import asyncio
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Telemetry ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("CVPR_NeuroSymbolic")

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

logger.info(f"Loading {MODEL_ID} into VRAM (bfloat16)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
logger.info("14B Text Model loaded successfully.")

app = FastAPI(title="Neuro-Symbolic Mosaics")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class MosaicRequest(BaseModel):
    prompt: str
    rows: int
    cols: int
    temperature: float = 0.4

# --- Vector to Raster (V2R) Engine ---
class Canvas:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = [["#000000" for _ in range(cols)] for _ in range(rows)]

    def _is_valid(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def fill(self, color):
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid[r][c] = color

    def set_pixel(self, r, c, color):
        if self._is_valid(int(r), int(c)):
            self.grid[int(r)][int(c)] = color

    def rect(self, r, c, h, w, color):
        for i in range(int(r), int(r + h)):
            for j in range(int(c), int(c + w)):
                self.set_pixel(i, j, color)

    def circle(self, cr, cc, radius, color):
        cr, cc, radius = int(cr), int(cc), float(radius)
        for r in range(self.rows):
            for c in range(self.cols):
                if (r - cr)**2 + (c - cc)**2 <= radius**2:
                    self.set_pixel(r, c, color)

    def line(self, r0, c0, r1, c1, color):
        r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
        length = max(abs(r1 - r0), abs(c1 - c0))
        if length == 0:
            self.set_pixel(r0, c0, color)
            return
        for i in range(length + 1):
            t = i / length
            r = round(r0 * (1 - t) + r1 * t)
            c = round(c0 * (1 - t) + c1 * t)
            self.set_pixel(r, c, color)

    def triangle(self, r1, c1, r2, c2, r3, c3, color):
        """Barycentric coordinate rasterization to prevent API hallucination failures."""
        min_r, max_r = max(0, math.floor(min(r1, r2, r3))), min(self.rows - 1, math.ceil(max(r1, r2, r3)))
        min_c, max_c = max(0, math.floor(min(c1, c2, c3))), min(self.cols - 1, math.ceil(max(c1, c2, c3)))

        def sign(p1r, p1c, p2r, p2c, p3r, p3c):
            return (p1c - p3c) * (p2r - p3r) - (p2c - p3c) * (p1r - p3r)

        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                d1 = sign(r, c, r1, c1, r2, c2)
                d2 = sign(r, c, r2, c2, r3, c3)
                d3 = sign(r, c, r3, c3, r1, c1)
                has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
                has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
                if not (has_neg and has_pos):
                    self.set_pixel(r, c, color)

def generate_blocking(inputs, max_tokens, temp):
    with torch.no_grad():
        return model.generate(
            **inputs,
            max_new_tokens=max_tokens, 
            temperature=temp if temp > 0.05 else 0.05,
            do_sample=temp > 0.05,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

@app.get("/gpu_stats")
def get_gpu_stats():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        return {
            "name": torch.cuda.get_device_name(device),
            "total_gb": round(torch.cuda.get_device_properties(device).total_memory / (1024**3), 2),
            "allocated_gb": round(torch.cuda.memory_allocated(device) / (1024**3), 2),
            "free_gb": round((torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_reserved(device)) / (1024**3), 2)
        }
    return {"name": "CPU", "total_gb": 0, "allocated_gb": 0, "free_gb": 0}

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    file_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="index.html not found.")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/generate_mosaic")
async def generate_mosaic(req: MosaicRequest):
    system_prompt = (
        "You are an expert generative artist writing Python code to render pixel art. "
        "You have access to a `canvas` object with ONLY the following methods:\n"
        "  canvas.rows, canvas.cols (integers representing grid dimensions)\n"
        "  canvas.fill(color_hex)\n"
        "  canvas.set_pixel(r, c, color_hex)\n"
        "  canvas.rect(r, c, h, w, color_hex)\n"
        "  canvas.circle(cr, cc, radius, color_hex)\n"
        "  canvas.line(r0, c0, r1, c1, color_hex)\n"
        "  canvas.triangle(r1, c1, r2, c2, r3, c3, color_hex)\n\n"
        "INSTRUCTIONS:\n"
        "1. Write a Python function `def render(canvas):` that draws the requested subject.\n"
        "2. Use relative math (e.g., `canvas.rows // 2`) to scale the drawing dynamically.\n"
        "3. Do NOT use any methods outside the provided API list. Do NOT import external libraries.\n"
        "4. Output ONLY valid Python code inside a ```python block."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Write the `render(canvas)` function to draw: '{req.prompt}'. The grid is {req.rows}x{req.cols}."}
    ]
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    t0 = time.perf_counter()
    outputs = await asyncio.to_thread(generate_blocking, inputs, 750, req.temperature)
    t1 = time.perf_counter()
    
    raw_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
    logger.info(f"Generated python logic in {t1-t0:.2f}s")
    
    # Extract Python code
    code_match = re.search(r'```python\n(.*?)\n```', raw_output, re.DOTALL | re.IGNORECASE)
    python_code = code_match.group(1).strip() if code_match else raw_output

    # --- Secure Restricted Execution Environment ---
    canvas_instance = Canvas(req.rows, req.cols)
    
    # Safely expose standard built-ins required for spatial math/looping
    safe_builtins = {
        'range': range, 'int': int, 'float': float, 'round': round,
        'max': max, 'min': min, 'abs': abs, 'len': len, 'enumerate': enumerate, 'list': list
    }
    
    # Pre-inject `canvas` to protect against flat-script (non-function) hallucinations
    safe_globals = {
        "__builtins__": safe_builtins,
        "math": math,
        "canvas": canvas_instance
    }
    safe_locals = {}

    try:
        # Compile and execute the LLM's AST
        exec(python_code, safe_globals, safe_locals)
        
        # Check execution modes
        if "render" in safe_locals:
            safe_locals["render"](canvas_instance)
        elif "render" in safe_globals:
            safe_globals["render"](canvas_instance)
        else:
            # Code executed flatly against the globally injected `canvas`
            pass
            
    except Exception as e:
        logger.error(f"LLM Code Execution Failed: {e}")
        # Rendering failure heuristic: Draw a visual Error "X"
        canvas_instance.fill("#3b0707") # Dark red
        canvas_instance.line(0, 0, req.rows-1, req.cols-1, "#ff0000")
        canvas_instance.line(0, req.cols-1, req.rows-1, 0, "#ff0000")
        python_code = f"# COMPILE ERROR DETECTED:\n# {str(e)}\n\n" + python_code

    return {
        "code": python_code,
        "matrix": canvas_instance.grid,
        "raw_output": raw_output
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8123)
