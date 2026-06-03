import re
import fitz  # PyMuPDF
import jieba
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin

# Suppress jieba logging
import logging
logging.getLogger('jieba').setLevel(logging.ERROR)

STOP_WORDS = {
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都',
    '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会',
    '着', '没有', '看', '好', '自己', '这', '那', '它', '他', '她',
    '这个', '那个', '如果', '因为', '所以', '但是', '然后', '可以',
    '能', '对', '与', '以', '及', '把', '将', '被', '通过', '按照',
    '进行', '使用', '需要', '具有', '如下', '如图', '实验',
    '报告', '问题', '描述', '方法', '结果', '分析', '图', '表',
}


IMG_FEATURE_KEYS = ['page_count', 'img_count', 'img_area_ratio', 'imgs_per_page']


class TextSelector(BaseEstimator, TransformerMixin):
    """从 (text, img_feat) 元组中取出文字部分，供 Pipeline 使用。"""
    def fit(self, X, y=None): return self
    def transform(self, X): return [x[0] for x in X]


class ImgFeatureSelector(BaseEstimator, TransformerMixin):
    """从 (text, img_feat) 元组中取出图片结构特征，转为 numpy 数组。"""
    def fit(self, X, y=None): return self
    def transform(self, X):
        return np.array([[x[1][k] for k in IMG_FEATURE_KEYS] for x in X])


def jieba_tokenizer(text: str) -> list:
    words = jieba.cut(text)
    return [w for w in words if w.strip() and w not in STOP_WORDS and len(w) > 1]


def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return '\n'.join(pages)


def clean_text(text: str) -> str:
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    # Keep Chinese, alphanumerics, basic punctuation
    text = re.sub(r'[^一-鿿　-〿＀-￯\w\s]', ' ', text)
    return text.strip()


def extract_image_features(pdf_path: str) -> dict:
    """提取图片相关的结构特征（无需OCR）"""
    doc = fitz.open(pdf_path)
    page_count = len(doc)
    img_count = 0
    total_img_area = 0
    page_area = 0

    for page in doc:
        rect = page.rect
        page_area += rect.width * rect.height
        for img in page.get_images():
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                img_count += 1
                total_img_area += pix.width * pix.height
            except Exception:
                img_count += 1

    doc.close()
    img_area_ratio = total_img_area / page_area if page_area > 0 else 0
    return {
        'page_count': page_count,
        'img_count': img_count,
        'img_area_ratio': img_area_ratio,
        'imgs_per_page': img_count / page_count if page_count > 0 else 0,
    }


def preprocess_pdf(pdf_path: str) -> str:
    text = extract_text(pdf_path)
    return clean_text(text)


def load_dataset(train_dir: str = 'train'):
    train_path = Path(train_dir)
    texts, labels, paths, img_features = [], [], [], []
    for grade in ['A', 'B', 'C', 'D']:
        grade_dir = train_path / grade
        if not grade_dir.exists():
            continue
        for pdf_file in sorted(grade_dir.glob('*.pdf')):
            text = preprocess_pdf(str(pdf_file))
            if text.strip():
                texts.append(text)
                labels.append(grade)
                paths.append(str(pdf_file))
                img_features.append(extract_image_features(str(pdf_file)))
    return texts, labels, paths, img_features
