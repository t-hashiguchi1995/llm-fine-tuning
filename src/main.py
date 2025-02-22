"""
メインモジュール

このスクリプトは、Hugging Face Transformersを使用してBERTモデルのファインチューニングを行います。IMDbデータセットを使用し、映画レビューの感情分析モデルを作成します。
"""

from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    """
    モデル評価のためのメトリクスを計算する関数

    Args:
        eval_pred: 評価予測結果
    Returns:
        metrics: 精度、適合率、再現率、F1スコアを含む辞書
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

def tokenize_function(example):
    """
    トークナイズ関数

    Args:
        example: データセットの各例
    Returns:
        トークナイズされた入力
    """
    return tokenizer(example["text"], padding="max_length", truncation=True)

def main():
    """
    メイン関数
    """
    # モデル名の指定
    model_name = "bert-base-uncased"

    # トークナイザとモデルのロード
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # データセットのロード（IMDbレビュー）
    print("データセットをロードしています...")
    dataset = load_dataset("imdb")

    # データセットのトークナイズ
    print("データセットをトークナイズしています...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # トレーニングと評価用データセットの準備
    print("データセットを準備しています...")
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(2000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    # トレーニングパラメータの設定
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
    )

    # トレーナーの初期化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    # トレーニングの実行
    print("モデルのファインチューニングを開始します...")
    trainer.train()

    # モデルの評価
    print("モデルを評価しています...")
    eval_results = trainer.evaluate()
    print(f"評価結果: {eval_results}")

    # モデルの保存
    print("ファインチューニング済みモデルを保存しています...")
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    print("完了しました。")

if __name__ == "__main__":
    main()