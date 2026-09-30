# Cross-Domain Emotion Detection: Reddit to Customer Support

**How reliably can an NLP model trained on social media emotions identify fine-grained emotions in customer support messages?**

UCL MSc Business Analytics, Natural Language Processing (2025/26). Group project, 5 members.

**My contribution:** TF-IDF + Logistic Regression baseline, manual annotation of 250 tweets, cross-domain evaluation framework, and results write-up.

---

## Overview

Companies want to detect customer emotions (anger, confusion, gratitude) automatically to route and prioritise support tickets. Labelled customer support data is scarce, so a common shortcut is to train on a large public dataset from a different domain.

This project tests whether that shortcut works. We trained a baseline and a fine-tuned BERT model on **GoEmotions** (Reddit comments, 28 emotion classes), then evaluated both on **500 manually annotated Twitter customer support messages**.

## Key findings

- **BERT beats the baseline in-domain:** macro F1 of **0.434** vs **0.305** for TF-IDF + Logistic Regression, a 42% relative improvement.
- **Both models transfer poorly:** weighted F1 drops by about **25 to 26 percentage points** on Twitter data for both models. Because the gap is nearly identical, the bottleneck is the domain mismatch, not model capacity.
- **Coarse labels are more practical:** grouping the 28 emotions into positive, negative and ambiguous raises BERT's Twitter weighted F1 from **0.312 to 0.485**.
- **Annotation is reliable:** two annotators reached Cohen's κ = **0.733** (substantial agreement) on the 500 tweets.

## Results

| Model | In-domain (GoEmotions) weighted F1 | Cross-domain (Twitter) weighted F1 | Transfer gap |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.446 | 0.186 | 26.0 pp |
| BERT (fine-tuned) | 0.566 | 0.312 | 25.5 pp |
| BERT, grouped into 3 classes | | 0.485 | |

pp = percentage points.

## Business recommendations

1. Don't deploy fine-grained (28-class) emotion labels in production yet.
2. Use coarse sentiment (positive, negative, ambiguous) for ticket routing.
3. Collect in-domain labelled data before scaling.
4. Send low-confidence predictions to human review.
5. Track the emotion distribution over time as an early-warning signal.

## Method

1. **Exploration:** class distributions and text length differences between Reddit and Twitter.
2. **Baseline:** TF-IDF features + multinomial Logistic Regression (28 classes).
3. **BERT:** `bert-base-uncased` fine-tuned with the Hugging Face Trainer API (3 epochs).
4. **Annotation:** 500 customer support tweets labelled independently by two annotators with the same 28-emotion taxonomy, finalised by consensus.
5. **Cross-domain evaluation:** both models applied to the annotated tweets to measure the transfer gap.
6. **Error analysis:** confusion matrices, per-class F1 and misclassified examples.
7. **Grouped evaluation:** emotions collapsed into 3 groups as a practical fallback.

## Data

Data is not stored in this repo. The notebook loads it directly:

- **Training:** [GoEmotions](https://huggingface.co/datasets/go_emotions), simplified version (about 58,000 Reddit comments; 43,410 in the training split)
- **Evaluation:** [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (ThoughtVector), 500 tweets sampled and annotated by the team

## Repository structure

| File | Description |
|---|---|
| `emotion_transfer_goemotions_to_twitter.ipynb` | Main notebook: data loading, training, evaluation, error analysis, recommendations |
| `report.pdf` | Written report |
| `presentation.pdf` | Presentation slides |
| `LICENSE` | MIT licence |

## How to run

1. Open the notebook in Kaggle (recommended, GPU enabled) or Google Colab.
2. Add the Customer Support on Twitter dataset as an input.
3. Run all cells. The notebook installs packages and downloads GoEmotions automatically.

## Tech stack

Python, PyTorch, Hugging Face Transformers and Datasets, scikit-learn, pandas, NumPy, Matplotlib, Seaborn

## Team

| Member | Contribution |
|---|---|
| Leya Sherman | BERT fine-tuning, manual annotation (250 tweets), methodology writing |
| Marcus Hudson | Related works, error analysis writing |
| Majid Ahmadi Moghaddam | Exploratory data analysis, results visualisations |
| Pablo Williams | Data preprocessing pipeline, grouped emotion evaluation |
| Salman Khalid | TF-IDF baseline, manual annotation (250 tweets), cross-domain evaluation, results writing |

## Reference

Demszky, D. et al. (2020) 'GoEmotions: A Dataset of Fine-Grained Emotions', *Proceedings of ACL 2020*.
