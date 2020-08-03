from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
from newsqa import NewsQaModel, get_single_prediction
from transformers import BertTokenizer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# All the news stories in the dataset
NEWS_STORIES = pickle.load(open('data/news_stories.pkl', 'rb'))

# The base model and tokenizer to use
model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
tokenzer = BertTokenizer.from_pretrained(model_name)

# Load the finetuned model
newsqa_model = NewsQaModel()
newsqa_model.load('data/bert_model.pt')

class InputData(BaseModel):
    text: str
    question: str

@app.get("/article/{article_key}")
async def get_article(article_key):
    return {"article": NEWS_STORIES[article_key]}

@app.post("/")
async def get_answer(inputdata: InputData):
    ans_texts, char_ranges = get_single_prediction(inputdata.text, inputdata.question, 
                                                  tokenzer, newsqa_model, doc_stride = 512)
    
    return {"answer_texts": ans_texts, "char_ranges": char_ranges}