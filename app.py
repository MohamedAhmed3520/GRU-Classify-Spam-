import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re
import time

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

st.set_page_config(
    page_title="News Clickbait Detector",
    page_icon="📰",
    layout="wide"
)

for pkg in ["punkt_tab","stopwords","wordnet"]:
    nltk.download(pkg, quiet=True)

stop_words=set(stopwords.words("english"))
lemmatizer=WordNetLemmatizer()

st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#0f172a,#111827,#1e293b);
}

#MainMenu,footer,header{
visibility:hidden;
}

.title{
font-size:50px;
font-weight:800;
text-align:center;
color:#fb923c;
}

.subtitle{
text-align:center;
font-size:18px;
color:white;
margin-bottom:30px;
}

.card{
background:rgba(255,255,255,.05);
padding:20px;
border-radius:15px;
border:1px solid rgba(255,255,255,.1);
backdrop-filter:blur(15px);
}

.stButton>button{
width:100%;
height:55px;
font-size:18px;
font-weight:bold;
background:#ea580c;
color:white;
border-radius:12px;
}

.terminal{
background:#020617;
color:#22c55e;
padding:18px;
border-radius:15px;
font-family:monospace;
border:1px solid #22c55e;
white-space:pre-wrap;
}

.blob1{
position:fixed;
width:220px;
height:220px;
border-radius:50%;
background:#fb923c33;
top:-50px;
left:-60px;
filter:blur(30px);
animation:f1 8s infinite;
}

.blob2{
position:fixed;
width:180px;
height:180px;
border-radius:50%;
background:#ef444433;
bottom:-40px;
right:-40px;
filter:blur(25px);
animation:f2 10s infinite;
}

@keyframes f1{
50%{transform:translateY(-25px);}
}

@keyframes f2{
50%{transform:translateY(25px);}
}

</style>

<div class="blob1"></div>
<div class="blob2"></div>

<div class="title">

📰 Clickbait Detector

</div>

<div class="subtitle">

GRU + DistilBERT Tokenizer

</div>

""",unsafe_allow_html=True)

def preprocess_text(text):

    text=re.sub(r'[^A-Za-z\\s]','',text)

    text=re.sub(r'https?://\\S+|www\\.\\S+','',text)

    text=text.lower()

    tokens=word_tokenize(text)

    tokens=[
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)

class GRUClassifier(nn.Module):

    def __init__(self,vocab_size,embed_dim,hidden_dim,num_classes):

        super().__init__()

        self.embedding=nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        self.gru=nn.GRU(
            embed_dim,
            hidden_dim,
            batch_first=True
        )

        self.fc=nn.Linear(
            hidden_dim,
            num_classes
        )

    def forward(self,input_ids):

        x=self.embedding(input_ids)

        out,_=self.gru(x)

        out=out[:,-1,:]

        return self.fc(out)

device=torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)

@st.cache_resource
def load():

    tokenizer=DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )

    model=GRUClassifier(
        vocab_size=tokenizer.vocab_size,
        embed_dim=128,
        hidden_dim=64,
        num_classes=2
    )

    model.load_state_dict(
        torch.load(
            "best_gru_model.pth",
            map_location=device
        )
    )

    model.to(device)

    model.eval()

    return tokenizer,model

tokenizer,model=load()

left,right=st.columns([2,1])

with left:

    headline=st.text_area(

        "📝 Enter News Headline",

        height=170,

        placeholder="Type a headline..."

    )

with right:

    st.markdown("""

<div class="card">

<h3>🤖 Model</h3>

<b>Architecture</b>

GRU

<br><br>

<b>Tokenizer</b>

DistilBERT

<br><br>

<b>Classes</b>

<ul>

<li>Ham</li>

<li>Clickbait</li>

</ul>

</div>

""",unsafe_allow_html=True)

predict_btn=st.button(
"🚀 Analyze Headline"
)
# ==========================
# Prediction
# ==========================

if predict_btn:

    if headline.strip() == "":
        st.warning("⚠ Please enter a news headline.")
        st.stop()

    progress = st.progress(0)

    for i in range(100):
        progress.progress(i + 1)
        time.sleep(0.005)

    progress.empty()

    clean = preprocess_text(headline)

    encoding = tokenizer(
        clean,
        truncation=True,
        padding="max_length",
        max_length=32,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)

    with torch.no_grad():

        output = model(input_ids)

        probs = F.softmax(output, dim=1)[0]

    confidence, pred = torch.max(probs, 0)

    confidence = float(confidence) * 100

    prediction = (
        "📢 Clickbait"
        if pred.item() == 1
        else
        "✅ Ham"
    )

# ==========================
# AI Terminal
# ==========================

    terminal = st.empty()

    text = f"""

> SYSTEM ONLINE

> CLEANING TEXT...

> TOKENIZING...

> RUNNING GRU...

> PREDICTION:

{prediction}

> CONFIDENCE:

{confidence:.2f} %

> STATUS:

SUCCESS

"""

    current = ""

    for ch in text:

        current += ch

        terminal.markdown(
            f"""
<div class="terminal">

{current}

</div>
""",
            unsafe_allow_html=True
        )

        time.sleep(.002)

# ==========================
# Cards
# ==========================

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:

        if pred.item() == 1:

            color = "#ef4444"

            status = "HIGH CLICKBAIT RISK"

        else:

            color = "#22c55e"

            status = "SAFE HEADLINE"

        st.markdown(

            f"""

<div style="
background:#111827;
padding:25px;
border-radius:15px;
border-left:8px solid {color};
">

<h2>{prediction}</h2>

<h3>{confidence:.2f}%</h3>

<p>{status}</p>

</div>

""",

            unsafe_allow_html=True

        )

# ==========================
# Confidence Table
# ==========================

        df = pd.DataFrame({

            "Class": [

                "Ham",

                "Clickbait"

            ],

            "Probability (%)": [

                float(probs[0]) * 100,

                float(probs[1]) * 100

            ]

        })

        st.dataframe(

            df,

            use_container_width=True,

            hide_index=True

        )

# ==========================
# Chart
# ==========================

    with c2:

        fig, ax = plt.subplots(figsize=(6,4))

        ax.barh(

            ["Ham", "Clickbait"],

            [

                float(probs[0]) * 100,

                float(probs[1]) * 100

            ]

        )

        ax.set_xlim(0,100)

        ax.set_xlabel("Confidence (%)")

        st.pyplot(fig)

# ==========================
# Footer
# ==========================

st.markdown("---")

st.markdown("""

<center>

<h4 style="color:#94a3b8">

📰 Clickbait Detector • GRU • PyTorch • Streamlit

</h4>

</center>

""", unsafe_allow_html=True)