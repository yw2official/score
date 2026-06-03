"""
批量预测测试集，输出 CSV。

用法：
  python predict_batch.py <测试目录> [选项]

目录结构支持两种：
  有标签（用于评估）：   test/A/*.pdf  test/B/*.pdf  ...
  无标签（纯预测）：     test/*.pdf

选项：
  --mode   tfidf | bert | ensemble   默认 tfidf
  --out    输出 CSV 路径             默认 results.csv
  --bert-weight  集成模式中 BERT 的权重  默认 0.4

示例：
  python predict_batch.py test/ --mode tfidf
  python predict_batch.py test/ --mode bert   --out bert_results.csv
  python predict_batch.py test/ --mode ensemble
"""
import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

from preprocess import preprocess_pdf, extract_image_features, TextSelector, ImgFeatureSelector

MODEL_PATH = 'models/model.pkl'
BERT_PATH = 'models/bert'
GRADES = ['A', 'B', 'C', 'D']

# ── 模型加载（懒加载，只在需要时加载）─────────────────────────────────────────

_tfidf_model = None
_bert_model = None


def _get_tfidf():
    global _tfidf_model
    if _tfidf_model is None:
        if not Path(MODEL_PATH).exists():
            sys.exit(f"[错误] 未找到 TF-IDF 模型: {MODEL_PATH}，请先运行 python train.py")
        with open(MODEL_PATH, 'rb') as f:
            _tfidf_model = pickle.load(f)
    return _tfidf_model


def _get_bert():
    global _bert_model
    if _bert_model is None:
        bert_pkl = Path(BERT_PATH) / 'bert_classifier.pkl'
        if not bert_pkl.exists():
            sys.exit(f"[错误] 未找到 BERT 模型: {bert_pkl}，请先运行 python train_bert.py")
        with open(bert_pkl, 'rb') as f:
            _bert_model = pickle.load(f)
    return _bert_model


# ── 单文件预测 ─────────────────────────────────────────────────────────────────

def _predict_tfidf(text, img_feat):
    model = _get_tfidf()
    X = [(text, img_feat)]
    grade = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    classes = model.classes_
    return grade, {c: float(p) for c, p in zip(classes, probs)}


def _predict_bert(text):
    bert = _get_bert()
    probs = bert.predict_proba([text])[0]
    classes = bert.classes_
    grade = classes[probs.argmax()]
    return grade, {c: float(p) for c, p in zip(classes, probs)}


def _predict_ensemble(text, img_feat, bert_weight=0.4):
    from claude_processor import grade_with_claude
    _, bert_probs = _predict_bert(text)
    claude_result = grade_with_claude(text)
    claude_probs = claude_result['probabilities']
    cw = bert_weight
    merged = {g: round(cw * bert_probs.get(g, 0) + (1 - cw) * claude_probs.get(g, 0), 4)
              for g in GRADES}
    total = sum(merged.values())
    merged = {g: round(v / total, 4) for g, v in merged.items()}
    grade = max(merged, key=merged.get)
    return grade, merged


def predict_pdf(pdf_path: str, mode: str, bert_weight: float = 0.4):
    text = preprocess_pdf(pdf_path)
    img_feat = extract_image_features(pdf_path)
    if mode == 'tfidf':
        return _predict_tfidf(text, img_feat)
    elif mode == 'bert':
        return _predict_bert(text)
    elif mode == 'ensemble':
        return _predict_ensemble(text, img_feat, bert_weight)
    else:
        raise ValueError(f"未知模式: {mode}")


# ── 收集 PDF 文件 ──────────────────────────────────────────────────────────────

def collect_pdfs(root: Path):
    """
    返回 [(pdf_path, true_label_or_None), ...]
    如果目录下有 A/B/C/D 子目录，读取标签；否则 label=None。
    """
    entries = []
    labeled = any((root / g).is_dir() for g in GRADES)
    if labeled:
        for g in GRADES:
            grade_dir = root / g
            if grade_dir.exists():
                for pdf in sorted(grade_dir.glob('*.pdf')):
                    entries.append((pdf, g))
    else:
        for pdf in sorted(root.glob('*.pdf')):
            entries.append((pdf, None))
    return entries


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='批量预测作业评分')
    parser.add_argument('test_dir', help='测试集目录')
    parser.add_argument('--mode', default='tfidf', choices=['tfidf', 'bert', 'ensemble'])
    parser.add_argument('--out', default='results.csv', help='输出 CSV 路径')
    parser.add_argument('--bert-weight', type=float, default=0.4,
                        help='集成模式中 BERT 的权重 (默认 0.4)')
    args = parser.parse_args()

    root = Path(args.test_dir)
    if not root.exists():
        sys.exit(f"[错误] 目录不存在: {root}")

    entries = collect_pdfs(root)
    if not entries:
        sys.exit(f"[错误] {root} 下没有找到 PDF 文件")

    has_labels = entries[0][1] is not None
    print(f"找到 {len(entries)} 个 PDF，模式: {args.mode}，{'有' if has_labels else '无'}真实标签")
    print(f"输出: {args.out}\n")

    correct = 0
    rows = []

    for i, (pdf_path, true_label) in enumerate(entries, 1):
        t0 = time.time()
        try:
            grade, probs = predict_pdf(str(pdf_path), args.mode, args.bert_weight)
            elapsed = time.time() - t0
            match = (grade == true_label) if true_label else None
            if match is True:
                correct += 1
            status = '✓' if match is True else ('✗' if match is False else ' ')
            true_str = true_label or '-'
            print(f"[{i:3d}/{len(entries)}] {status} {pdf_path.name:40s} "
                  f"预测:{grade}  真实:{true_str}  "
                  f"({elapsed:.1f}s)")
            rows.append({
                'file': str(pdf_path),
                'filename': pdf_path.name,
                'true_grade': true_label or '',
                'pred_grade': grade,
                'correct': '' if true_label is None else str(grade == true_label),
                'confidence': f"{max(probs.values()):.4f}",
                'prob_A': f"{probs.get('A', 0):.4f}",
                'prob_B': f"{probs.get('B', 0):.4f}",
                'prob_C': f"{probs.get('C', 0):.4f}",
                'prob_D': f"{probs.get('D', 0):.4f}",
            })
        except Exception as e:
            print(f"[{i:3d}/{len(entries)}] ! {pdf_path.name:40s} 错误: {e}")
            rows.append({
                'file': str(pdf_path), 'filename': pdf_path.name,
                'true_grade': true_label or '', 'pred_grade': 'ERROR',
                'correct': '', 'confidence': '',
                'prob_A': '', 'prob_B': '', 'prob_C': '', 'prob_D': '',
            })

    # 写 CSV
    fieldnames = ['file', 'filename', 'true_grade', 'pred_grade',
                  'correct', 'confidence', 'prob_A', 'prob_B', 'prob_C', 'prob_D']
    with open(args.out, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*60}")
    if has_labels:
        labeled_rows = [r for r in rows if r['true_grade'] and r['pred_grade'] != 'ERROR']
        acc = correct / len(labeled_rows) if labeled_rows else 0
        print(f"准确率: {correct}/{len(labeled_rows)} = {acc:.4f}")

        # 按类别统计
        from collections import defaultdict
        per_class = defaultdict(lambda: {'correct': 0, 'total': 0})
        for r in labeled_rows:
            g = r['true_grade']
            per_class[g]['total'] += 1
            if r['correct'] == 'True':
                per_class[g]['correct'] += 1
        print("\n各等级准确率:")
        for g in GRADES:
            s = per_class[g]
            if s['total']:
                print(f"  {g}: {s['correct']}/{s['total']} = {s['correct']/s['total']:.4f}")

    print(f"\n结果已保存至: {args.out}")


if __name__ == '__main__':
    main()
