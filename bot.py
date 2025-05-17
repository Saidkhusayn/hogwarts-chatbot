import os
import logging
import json

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import cohere

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ParseMode
from dotenv import load_dotenv
load_dotenv()

# === CONFIG ===
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN")
COHERE_API_KEY  = os.getenv("COHERE_API_KEY")
CSV_PATH        = "data.csv"

EMBED_MODEL     = "all-MiniLM-L6-v2"
TOP_K           = 5
COHERE_MODEL    = "command-r-plus"
COHERE_MAX_TOKS = 300

# === LOGGING ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === LOAD HIERARCHY FOR MENU ===
df = pd.read_csv(CSV_PATH)
hierarchy = {}
for _, row in df.iterrows():
    sec = row["Section"]
    sub = row["Sub-section"]
    content = row["Content"]
    raw_add = row.get("Additional", "")
    additional = "" if (isinstance(raw_add, float) and pd.isna(raw_add)) else str(raw_add).strip()
    hierarchy.setdefault(sec, {})[sub] = {
        "content": content,
        "additional": additional
    }

# === LOAD CACHED CHUNKS & EMBEDDINGS ===
with open("chunks.json", "r") as f:
    chunks = json.load(f)
embs_np = np.load("embs.npy")

# === BUILD FAISS INDEX ===
d = embs_np.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embs_np)

# === LOAD LOCAL EMBEDDING MODEL ===
model = SentenceTransformer(EMBED_MODEL)

# === INIT COHERE CLIENT ===
co = cohere.Client(COHERE_API_KEY)

# === KEYBOARD BUILDER ===
def make_keyboard(items, prefix):
    buttons = [[InlineKeyboardButton(text=item, callback_data=f"{prefix}|{item}")] for item in items]
    return InlineKeyboardMarkup(buttons)

# === MENU HANDLERS ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = make_keyboard(list(hierarchy.keys()), prefix="SEC")
    await update.message.reply_text("Please choose a section:", reply_markup=keyboard)

async def section_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    _, section = query.data.split("|", 1)
    context.user_data["section"] = section

    subs = list(hierarchy[section].keys())
    keyboard = make_keyboard(subs, prefix="SUB")
    await query.edit_message_text(
        text=f"*Section:* {section}\n*Please choose one of the options:*",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=keyboard
    )

async def subsection_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    _, subsection = query.data.split("|", 1)
    section = context.user_data.get("section")

    data = hierarchy[section][subsection]
    content = data["content"]
    additional = data["additional"]

    await query.edit_message_text(
        f"*{section}* → *{subsection}*\n\n{content}",
        parse_mode=ParseMode.MARKDOWN
    )
    if additional:
        await query.message.reply_text(f"*More info:*\n{additional}", parse_mode=ParseMode.MARKDOWN)

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Operation cancelled.")

# === COHERE ANSWERING ===
async def answer_free_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text

    # 1) Embed query locally
    q_emb = model.encode(question)
    q_np  = np.array([q_emb], dtype="float32")

    # 2) Retrieve top K
    _, I = index.search(q_np, TOP_K)
    top_chunks = [chunks[i] for i in I[0]]

    # 3) Use Cohere's Chat API with documents
    try:
        response = co.chat(
            model=COHERE_MODEL,
            message=question,
            documents=[{"text": chunk} for chunk in top_chunks],
            temperature=0.3,
            max_tokens=COHERE_MAX_TOKS,
        )
        answer = response.text.strip()
    except Exception as e:
        logger.error(f"Cohere error: {e}")
        answer = "❌ Failed to generate a response from AI."

    # 4) Reply
    await update.message.reply_text(answer)

# === MAIN ===
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Menu navigation
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(section_chosen, pattern=r"^SEC\|"))
    app.add_handler(CallbackQueryHandler(subsection_chosen, pattern=r"^SUB\|"))
    app.add_handler(CommandHandler("cancel", cancel))

    # Free-text handler
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, answer_free_text))

    logger.info("Bot starting…")
    app.run_polling()

if __name__ == "__main__":
    main()