# NLP Sentiment Analysis: Cross-Domain Emotion Detection

**How reliably can an NLP model trained on social media emotions identify fine-grained emotions in customer support messages?**

> UCL MSIN0221: Natural Language Processing — Group Assignment (Team 1)

## Overview

This project investigates whether emotion detection models trained on Reddit data (GoEmotions dataset) can transfer effectively to classify emotions in Twitter customer support messages. We train a baseline Logistic Regression model and a fine-tuned BERT model on 28 fine-grained emotion classes, then evaluate cross-domain performance on 500 manually annotated customer support tweets.

## Key Findings

- **BERT fine-tuned on GoEmotions** achieves a macro F1 of **0.434** in-domain, a 42% improvement over the TF-IDF + Logistic Regression baseline (0.305)
- **Cross-domain transfer is poor** — both models lose ~25–26% weighted F1 when applied to Twitter customer support data, indicating domain mismatch dominates over model capacity
- **Coarse-grained sentiment (3 classes)** substantially outperforms fine-grained (28 classes) in cross-domain settings, making it more practical for production use
- **Inter-annotator agreement** on manually labelled Twitter data: Cohen's κ = 0.733 (substantial)

## Models

| Model | In-Domain (GoEmotions) | Cross-Domain (Twitter) | Transfer Gap |
|-------|------------------------|------------------------|--------------|
| TF-IDF + Logistic Regression | 0.446 weighted F1 | 0.186 weighted F1 | 26.0% |
| BERT (fine-tuned) | 0.566 weighted F1 | 0.312 weighted F1 | 25.5% |

## Datasets

- **Training:** [GoEmotions](https://huggingface.co/datasets/go_emotions) (simplified) — ~43,410 Reddit comments labelled with 28 emotions
- **Evaluation:** [Twitter Customer Support](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) — 500 tweets manually annotated by two team members using the same 28-emotion taxonomy

## Project Structure

| File | Description |
|------|-------------|
| `GP_team 1 (1).ipynb` | Main notebook (data loading, training, evaluation, error analysis) |
| `MSIN0221_team 1_final project.pdf` | Written report |
| `MSIN0221_team 1_final project_presentation (1).pptx` | Presentation slides |
| `MSIN0221 Assessment Brief - Group Assignment-6.pdf` | Assignment specification |
| `LICENSE` | MIT License |
| `README.md` | This file |


## Methodology

1. **Data exploration** — analyse class distributions and text length differences between Reddit and Twitter
2. **Baseline model** — TF-IDF vectorisation + multinomial Logistic Regression (28 classes)
3. **BERT fine-tuning** — `bert-base-uncased` with HuggingFace Trainer API (3 epochs)
4. **Manual annotation** — 500 Twitter customer support tweets labelled independently by two annotators, finalised by consensus
5. **Cross-domain evaluation** — apply both models to annotated Twitter data and measure transfer gap
6. **Error analysis** — confusion matrices, per-class F1 breakdown, qualitative misclassification examples
7. **Grouped evaluation** — collapse 28 emotions into positive / negative / ambiguous for a practical fallback

## Tech Stack

- Python, PyTorch, HuggingFace Transformers & Datasets
- scikit-learn (TF-IDF, Logistic Regression, evaluation metrics)
- pandas, NumPy, Matplotlib, Seaborn

## Team Contributions

| Member | Contribution |
|--------|-------------|
| Leya Sherman | Data preprocessing, BERT fine-tuning, model training |
| Marcus Hudson | Introduction, literature review, error analysis, discussion |
| Majid Ahmadi Moghaddam | Methodology write-up |
| Pablo Williams | Presentation slides |
| Salman Khalid | Baseline model, evaluation framework, results analysis |

## How to Run

1. Open `GP_team 1 (1).ipynb` in Google Colab or Kaggle
2. Run all cells sequentially — the notebook handles package installation and dataset loading automatically

## License

MIT License — see [LICENSE](LICENSE) for details.






