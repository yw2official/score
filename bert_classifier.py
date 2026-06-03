"""
Chinese BERT-based homework classifier.

Hardware auto-selection:
  GPU → fine-tune BERT end-to-end (BertFineTuned)
  CPU → BERT fixed feature extractor + SVM (BertSVMClassifier, faster)
"""
import pickle
import numpy as np
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import SVC

LABEL2ID = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
MAX_LENGTH = 512
BERT_MODEL_PATH = 'models/bert'
DEFAULT_MODEL_NAME = 'bert-base-chinese'

# Module-level cache: BERT model is large, load once per process
_bert_cache: dict = {}


def _load_bert_base(local_path: str):
    if local_path not in _bert_cache:
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        model = AutoModel.from_pretrained(local_path)
        model.eval()
        _bert_cache[local_path] = (tokenizer, model)
    return _bert_cache[local_path]


def _get_embeddings(texts: list, bert_path: str, batch_size: int = 8) -> np.ndarray:
    """Extract [CLS] token embeddings. Returns (N, hidden_size) array."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer, model = _load_bert_base(bert_path)
    model = model.to(device)

    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, truncation=True, padding=True,
            max_length=MAX_LENGTH, return_tensors='pt',
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            emb = out.last_hidden_state[:, 0, :].cpu().numpy()
            all_emb.append(emb)

        done = min(i + batch_size, len(texts))
        if done % 40 == 0 or done == len(texts):
            print(f"  embeddings {done}/{len(texts)}")

    return np.vstack(all_emb)


# ─── CPU mode ────────────────────────────────────────────────────────────────

class BertSVMClassifier:
    """BERT feature extractor + SVM. Recommended for CPU-only environments."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.bert_path = str(Path(BERT_MODEL_PATH) / 'pretrained')
        self.clf = SVC(
            kernel='rbf', C=10, gamma='scale',
            class_weight='balanced', probability=True, random_state=42,
        )

    def fit(self, texts: list, labels: list):
        if not Path(self.bert_path).exists():
            print(f"  Downloading {self.model_name} → {self.bert_path}")
            print("  提示: 若下载缓慢，可先 export HF_ENDPOINT=https://hf-mirror.com")
            tok = AutoTokenizer.from_pretrained(self.model_name)
            mdl = AutoModel.from_pretrained(self.model_name)
            Path(self.bert_path).mkdir(parents=True, exist_ok=True)
            tok.save_pretrained(self.bert_path)
            mdl.save_pretrained(self.bert_path)
            print("  已保存至本地")

        print("  提取 BERT embeddings...")
        embeddings = _get_embeddings(texts, self.bert_path)
        print("  训练 SVM...")
        self.clf.fit(embeddings, labels)
        return self

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        emb = _get_embeddings(texts, self.bert_path)
        return list(self.clf.predict(emb))

    def predict_proba(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        emb = _get_embeddings(texts, self.bert_path)
        return self.clf.predict_proba(emb)

    @property
    def classes_(self):
        return self.clf.classes_


# ─── GPU mode ────────────────────────────────────────────────────────────────

class BertFineTuned:
    """Fine-tuned BERT. Recommended when GPU is available."""

    class _DS(Dataset):
        def __init__(self, texts, labels, tokenizer):
            self.enc = tokenizer(
                texts, truncation=True, padding=True,
                max_length=MAX_LENGTH, return_tensors='pt',
            )
            self.labels = torch.tensor([LABEL2ID[l] for l in labels])

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.enc.items()} | {'labels': self.labels[idx]}

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.saved_path = str(Path(BERT_MODEL_PATH) / 'finetuned')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._tok = None
        self._mdl = None

    def fit(self, texts, labels, epochs=5, batch_size=8, lr=2e-5):
        from transformers import get_linear_schedule_with_warmup
        from sklearn.utils.class_weight import compute_class_weight

        print(f"  Fine-tuning BERT on {self.device}")
        tok = AutoTokenizer.from_pretrained(self.model_name)
        mdl = BertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=4,
            id2label=ID2LABEL, label2id=LABEL2ID,
        ).to(self.device)

        # Sqrt-balanced weights: still favour minority classes (C, D) but less aggressively.
        # Full balanced: A=0.58 B=0.73 C=1.37 D=5.83  → D dominates, A/B undertrained
        # Sqrt balanced: A=0.76 B=0.85 C=1.17 D=2.41  → gentler gradient, all classes learn
        label_ids = np.array([LABEL2ID[l] for l in labels])
        cw = compute_class_weight('balanced', classes=np.arange(4), y=label_ids)
        cw = np.sqrt(cw)
        cw = cw / cw.mean()   # keep mean=1 so overall loss scale is stable
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(cw, dtype=torch.float).to(self.device),
            label_smoothing=0.1,  # smooth A/B boundary: penalise overconfident predictions
        )
        print(f"  Class weights (sqrt-balanced): " +
              " ".join(f"{ID2LABEL[i]}={cw[i]:.2f}" for i in range(4)))

        loader = DataLoader(self._DS(texts, labels, tok),
                            batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.01)
        total = len(loader) * epochs
        sched = get_linear_schedule_with_warmup(optimizer, total // 10, total)

        mdl.train()
        for ep in range(epochs):
            loss_sum, correct = 0, 0
            for batch in loader:
                labels_batch = batch.pop('labels').to(self.device)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = mdl(**batch).logits          # no labels → no internal loss
                loss = loss_fn(logits, labels_batch)  # weighted CE
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                optimizer.step()
                sched.step()
                optimizer.zero_grad()
                loss_sum += loss.item()
                correct += (logits.argmax(-1) == labels_batch).sum().item()
            print(f"  Epoch {ep+1}/{epochs}: "
                  f"loss={loss_sum/len(loader):.4f}, acc={correct/len(texts):.4f}")

        Path(self.saved_path).mkdir(parents=True, exist_ok=True)
        mdl.save_pretrained(self.saved_path)
        tok.save_pretrained(self.saved_path)
        # Don't store model weights inside the pkl — they live in saved_path.
        # _ensure_loaded() reloads from disk on first inference call.
        self._tok, self._mdl = None, None
        return self

    def _ensure_loaded(self):
        if self._mdl is None:
            self._tok = AutoTokenizer.from_pretrained(self.saved_path)
            self._mdl = BertForSequenceClassification.from_pretrained(
                self.saved_path
            ).to(self.device)
            self._mdl.eval()

    def predict(self, texts):
        return [ID2LABEL[i] for i in self._probs(texts).argmax(axis=1)]

    def predict_proba(self, texts):
        return self._probs(texts)

    def _probs(self, texts):
        self._ensure_loaded()
        if isinstance(texts, str):
            texts = [texts]
        inputs = self._tok(texts, truncation=True, padding=True,
                           max_length=MAX_LENGTH, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._mdl(**inputs)
            return torch.softmax(out.logits, dim=-1).cpu().numpy()

    @property
    def classes_(self):
        return np.array(['A', 'B', 'C', 'D'])


# ─── Public API ──────────────────────────────────────────────────────────────

def train_bert(texts: list, labels: list,
               model_name: str = DEFAULT_MODEL_NAME,
               model_path: str = BERT_MODEL_PATH):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Hardware: {device}")

    if device.type == 'cuda':
        clf = BertFineTuned(model_name)
    else:
        clf = BertSVMClassifier(model_name)

    clf.fit(texts, labels)

    Path(model_path).mkdir(parents=True, exist_ok=True)
    save = f'{model_path}/bert_classifier.pkl'
    with open(save, 'wb') as f:
        pickle.dump(clf, f)
    print(f"  Saved → {save}")
    return clf


def load_bert_classifier(model_path: str = BERT_MODEL_PATH):
    with open(f'{model_path}/bert_classifier.pkl', 'rb') as f:
        return pickle.load(f)


def predict_bert(text: str, model_path: str = BERT_MODEL_PATH) -> tuple:
    """Returns (grade: str, probs: dict[str, float])."""
    clf = load_bert_classifier(model_path)
    probs = clf.predict_proba([text])[0]
    classes = clf.classes_
    pred = classes[probs.argmax()]
    return pred, {c: float(p) for c, p in zip(classes, probs)}
