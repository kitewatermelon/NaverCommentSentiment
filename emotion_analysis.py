import torch
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

emotion_labels = ['angry', 'anxious', 'embarrassed', 'happy', 'heartache', 'sad']

def analyze_emotion(text, tokenizer, model, device):
    if not isinstance(text, str) or text.strip() == "":
        return {label: None for label in emotion_labels}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0].cpu()
    return {label: float(probs[i]) for i, label in enumerate(emotion_labels)}

def run_emotion_analysis(df, comment_col, model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    tqdm.pandas()
    emotion_results = df[comment_col].progress_apply(lambda x: analyze_emotion(x, tokenizer, model, device))
    emotion_df = emotion_results.apply(pd.Series)
    df = pd.concat([df, emotion_df], axis=1)
    return df
