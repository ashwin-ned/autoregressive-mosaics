import re
import os
import time
import logging
import asyncio
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Telemetry ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("CVPR_ASCII")

# --- Native PyTorch Initialization ---
# Use Qwen/Qwen2.5-14B-Instruct or 
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

logger.info(f"Loading {MODEL_ID} into VRAM (bfloat16)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
logger.info("Model loaded successfully.")

app = FastAPI(title="Hierarchical ASCII Mosaics")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class MosaicRequest(BaseModel):
    prompt: str
    rows: int
    cols: int
    temperature: float = 0.3

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

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    file_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="index.html not found.")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/generate_mosaic")
async def generate_mosaic(req: MosaicRequest):
    """Zero-shot generation of semantic ASCII topology and color mapping."""
    
    system_prompt = (
        "You are an expert artist who performs image synthesizing using mosaics. You must create a 2D mosaic of the user's prompt using a 2-step process.\n\n"
        "PHASE 1 (Semantic Topology):\n"
        f"Draw the subject using an ASCII grid of EXACTLY {req.rows} rows and {req.cols} columns. "
        "You MUST construct the geometry using ONLY this 5-character topological vocabulary:\n"
        "'.' : Background (Negative space / $\\Omega^c$)\n"
        "'#' : Boundary (Outer silhouettes and hard structural edges / $\\partial\\Omega$)\n"
        "'@' : Core Volume (Dense interior mass / Lambertian diffuse base / $\\Omega$)\n"
        "'~' : Shadows / Gradients (Depth, self-occlusion, and curvature)\n"
        "'*' : Highlights / Details (Specular focal points and fine high-frequency features)\n"
        "Wrap the grid strictly inside <ascii>...</ascii> tags.\n\n"
        "PHASE 2 (Color Mapping):\n"
        "Assign a 7-character HEX color code to EVERY ASCII character used.\n"
        "Wrap the mapping strictly inside <palette>...</palette> tags.\n"
        "Format exactly as: <palette>.:#000000, #:#111111, @:#FF0000, ~:#880000, *:#FFFFFF</palette>"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Subject: '{req.prompt}'.\nGrid: {req.rows}x{req.cols}.\nGenerate the ASCII mask and Palette."}
    ]
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    logger.info(f"Computing semantic topology for {req.rows}x{req.cols} grid...")
    t0 = time.perf_counter()
    
    # Token Budget: ASCII Matrix (~ M*N tokens) + Palette (~ 50 tokens) + padding
    max_tok = (req.rows * req.cols) + 150
    outputs = await asyncio.to_thread(generate_blocking, inputs, max_tok, req.temperature)
    
    t1 = time.perf_counter()
    raw_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
    logger.info(f"Inference complete in {t1-t0:.2f}s")
    
    # --- Robust Extraction ---
    # Extract ASCII Grid
    ascii_match = re.search(r'<ascii>\n?(.*?)\n?</ascii>', raw_output, re.DOTALL | re.IGNORECASE)
    ascii_grid = ascii_match.group(1).strip() if ascii_match else ""
    
    # Clean the grid to strict MxN dimensions
    lines = [re.sub(r'\s+', '', line) for line in ascii_grid.split('\n') if line.strip()]
    cleaned_grid = []
    for i in range(req.rows):
        if i < len(lines):
            row = lines[i][:req.cols].ljust(req.cols, '.')
        else:
            row = '.' * req.cols
        cleaned_grid.append(list(row))
        
    # Extract Palette
    palette = {".": "#000000"} # Background fallback
    pal_match = re.search(r'<palette>(.*?)</palette>', raw_output, re.DOTALL | re.IGNORECASE)
    if pal_match:
        # Match pattern: Char : Hex
        pairs = re.findall(r'([^\s:])\s*:\s*(#[0-9a-fA-F]{6})', pal_match.group(1))
        for char, color in pairs:
            palette[char] = color.upper()
            
    # Compile Final RGB Matrix
    hex_matrix = []
    for row in cleaned_grid:
        hex_row = [palette.get(char, "#000000") for char in row]
        hex_matrix.append(hex_row)
            
    return {
        "ascii_grid": cleaned_grid,
        "palette": palette,
        "hex_matrix": hex_matrix,
        "raw_output": raw_output
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8123)