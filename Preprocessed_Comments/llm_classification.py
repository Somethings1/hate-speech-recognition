import pandas as pd
import json
import os
import math
import time
from tqdm import tqdm
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field
import google.genai as genai

# --- SETUP ---
print("Clearing proxy vars...")
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
print("✅ Proxy cleared.")

try:
    API_KEY = "AIzaSyDeTahi999DJextlj6VXxdlS88W0m0bQbM"
    client = genai.Client(api_key=API_KEY)
    MODEL_NAME = "gemini-2.5-flash"
    print("✅ Gemini client ready.")
except Exception as e:
    raise RuntimeError(f"Gemini init failed: {e}")

tqdm.pandas(desc="Step 1: Creating payloads")

# --- LOAD DATA ---
print("\nLoading data...")

cache_file = 'video_context_cache.json'
dataframe_file = 'all_comments_loaded.parquet'
OUTPUT_FOLDER = "classified_comments_256"

# --- Load context cache ---
VIDEO_CONTEXT_CACHE = json.load(open(cache_file, 'r', encoding='utf-8')) if os.path.exists(cache_file) else {}
print(f"Loaded context cache: {len(VIDEO_CONTEXT_CACHE)} items.")

# --- Load dataframe ---
if os.path.exists(dataframe_file):
    df_all_comments = pd.read_parquet(dataframe_file)
else:
    raise FileNotFoundError("!!! DataFrame file not found.")

# --- Model Schemas ---
class CommentClassification(BaseModel):
    comment_id: str
    classification: str
    confidence: float
    justification: str

class ClassificationList(BaseModel):
    results: List[CommentClassification]

def call_gemini_api(prompt: str) -> List[Dict]:
    """Call Gemini API with structured output."""
    SYSTEM_INSTRUCTION = """
    You are an expert Vietnamese content moderator. Your task is to classify TikTok comments as hate speech or not.
    Hate speech = insults, degradation, or attacks toward people/groups.
    """
    generation_config = genai.types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ClassificationList,
        system_instruction=SYSTEM_INSTRUCTION
    )
    contents = [{"role": "user", "parts": [{"text": prompt}]}]

    try:
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=generation_config
        )
        return json.loads(resp.text).get("results", [])
    except Exception as e:
        print(f"⚠️ API error: {e}")
        return []

# --- Create payloads ---
def create_comment_payload(row):
    vid = str(row["video_id"])
    video_context = VIDEO_CONTEXT_CACHE.get(vid, "Context Not Found.")
    text = row.get("clean_text") if pd.notna(row.get("clean_text")) else row.get("comment_text")
    return json.dumps({
        "comment_id": str(row["comment_id"]),
        "video_context": video_context,
        "commenter_username": row["username"],
        "comment_content": text,
    }, ensure_ascii=False)

df_all_comments["comment_json"] = df_all_comments.progress_apply(create_comment_payload, axis=1)

# --- Skip already processed ---
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
done_ids = set()
for f in os.listdir(OUTPUT_FOLDER):
    if f.endswith(".csv"):
        try:
            df_done = pd.read_csv(os.path.join(OUTPUT_FOLDER, f), usecols=["comment_id"])
            done_ids.update(df_done["comment_id"].astype(str).tolist())
        except Exception:
            pass

df_to_process = df_all_comments[~df_all_comments["comment_id"].astype(str).isin(done_ids)].copy()
print(f"Processing {len(df_to_process):,} comments (skipping {len(done_ids):,}).")

# --- Worker function ---
def process_batch(batch_index, batch_data):
    output_file = os.path.join(OUTPUT_FOLDER, f"classified_batch_{batch_index:04d}.csv")
    if os.path.exists(output_file) and os.path.getsize(output_file) > 50:
        return f"Batch {batch_index} skipped (exists)."

    comment_json_list = ",\n".join(batch_data["comment_json"].tolist())
    user_prompt = f"""
    Please classify the following {len(batch_data)} TikTok comments based on the provided rules.
    ---START---
    [
    {comment_json_list}
    ]
    ---END---
    """

    results = call_gemini_api(user_prompt)
    if not results:
        return f"Batch {batch_index} failed."

    df_results = pd.DataFrame(results).rename(columns={
        "classification": "llm_hate_speech",
        "confidence": "llm_confidence",
        "justification": "llm_justification"
    })
    df_results["comment_id"] = df_results["comment_id"].astype(str)
    batch_data["comment_id"] = batch_data["comment_id"].astype(str)

    df_chunk = pd.merge(
        batch_data.drop(columns=["comment_json"]),
        df_results,
        on="comment_id",
        how="left"
    )
    df_chunk["video_context"] = df_chunk["video_id"].astype(str).map(VIDEO_CONTEXT_CACHE)
    final_cols = [
        "comment_id", "video_id", "username", "comment_text", "clean_text",
        "digg_count", "create_time", "video_context",
        "llm_hate_speech", "llm_confidence", "llm_justification"
    ]
    df_chunk = df_chunk[[c for c in final_cols if c in df_chunk.columns]]
    df_chunk.to_csv(output_file, index=False)
    return f"Batch {batch_index} done ({len(df_chunk)} comments)."

# --- Parallel run ---
BATCH_SIZE = 256
num_batches = math.ceil(len(df_to_process) / BATCH_SIZE)
MAX_WORKERS = 5  # tweak depending on API rate limit

print(f"\nRunning {num_batches} batches using {MAX_WORKERS} parallel threads...")
futures = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    for i in range(num_batches):
        start, end = i * BATCH_SIZE, (i + 1) * BATCH_SIZE
        batch_data = df_to_process.iloc[start:end].copy()
        if batch_data.empty:
            continue
        futures.append(executor.submit(process_batch, i + 1, batch_data))

    for f in tqdm(as_completed(futures), total=len(futures)):
        print(f.result())

print("\n✅ All parallel batches complete.")

