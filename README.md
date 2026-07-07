# 📰 GRU Spam & Ham Text Classification

A deep learning text classification project that uses a custom **GRU (Gated Recurrent Unit)** neural network built with **PyTorch** to classify text as either **Spam (Clickbait)** or **Ham (Non-Clickbait)**.

The project includes a modern **Streamlit** web application that allows users to enter a news headline or text and receive an instant prediction with confidence scores.

---

## 🚀 Live Demo

👉 *Add your Streamlit deployment link here*

---

## ✨ Features

- 🧠 Custom GRU architecture built with PyTorch
- 📰 Binary text classification (Spam / Ham)
- 🔤 DistilBERT tokenizer for text encoding
- 🧹 Text preprocessing using NLTK
- 📊 Confidence score visualization
- 🌐 Interactive Streamlit web application
- ⚡ Fast CPU inference

---

## 🛠️ Tech Stack

- Python
- PyTorch
- Streamlit
- Transformers (DistilBERT Tokenizer)
- NLTK
- Pandas
- Matplotlib

---

## 📂 Project Structure

```text
GRU-Spam-Classification/
│
├── app.py
├── best_gru_model.pth
├── requirements.txt
├── README.md
└── notebooks/
```

---

## 🧠 Model Architecture

The classifier consists of:

- Embedding Layer
- GRU Layer
- Fully Connected Layer
- Softmax Output

```text
Input Text
     │
Preprocessing
     │
DistilBERT Tokenizer
     │
Embedding Layer
     │
GRU
     │
Linear Layer
     │
Spam / Ham
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/GRU-Spam-Classification.git
cd GRU-Spam-Classification
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 💾 Model Saving

The trained model is saved using PyTorch's `state_dict`:

```python
torch.save(model.state_dict(), "best_gru_model.pth")
```

The model weights are loaded during inference using:

```python
model.load_state_dict(torch.load("best_gru_model.pth"))
model.eval()
```

---

## 📈 Model Performance

| Metric | Value |
|---------|------:|
| Model | GRU |
| Framework | PyTorch |
| Classes | 2 |
| Task | Binary Text Classification |
| Tokenizer | DistilBERT |
| Output | Spam / Ham |

> Replace this section with your final accuracy, precision, recall, and F1-score if available.

---

## 📌 Future Improvements

- Improve classification accuracy with additional data
- Experiment with BiGRU and LSTM models
- Add attention mechanism
- Deploy using Docker
- Support batch predictions

---

## 📄 License

This project is intended for educational and research purposes.

---

## 👨‍💻 Author

**Mohamed Ahmed**

If you found this project useful, consider giving it a ⭐ on GitHub!
