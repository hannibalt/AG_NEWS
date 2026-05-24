import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# 1. VERİ KÜMESİNİN YÜKLENMESİ
print("AG News veri kümesi yükleniyor...")
dataset = load_dataset("ag_news")

# Hızlı test/eğitim için veriyi pandas dataframe'e çevirelim
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

# Sınıf isimleri (0: World, 1: Sports, 2: Business, 3: Sci/Tech)
target_names = ["World", "Sports", "Business", "Sci/Tech"]

# --- YAKLAŞIM A: RANDOM FOREST & TF-IDF ---
def run_random_forest():
    print("\n=== RANDOM FOREST + TF-IDF MODELİ BAŞLATIYOR ===")
    
    # TF-IDF Vektörleştirme
    print("Metinler vektörleştiriliyor (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train = vectorizer.fit_transform(train_df['text'])
    X_test = vectorizer.transform(test_df['text'])
    
    y_train = train_df['label']
    y_test = test_df['label']
    
    # Model Eğitimi
    print("Random Forest modeli eğitiliyor (Bu işlem birkaç dakika sürebilir)...")
    rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Tahmin ve Değerlendirme
    print("Model değerlendiriliyor...")
    preds = rf_model.predict(X_test)
    
    print("\n[Random Forest Başarı Raporu]")
    print(classification_report(y_test, preds, target_names=target_names))
    print(f"Genel Doğruluk (Accuracy): {accuracy_score(y_test, preds):.4f}")


# --- YAKLAŞIM B: DistilBERT (DEAL FOR GPU) ---
# Not: Bu kısmı çalıştırmak için CUDA destekli bir GPU önerilir.
def run_distilbert():
    print("\n=== DistilBERT MODELİ BAŞLATIYOR ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Çalıştırılan Cihaz: {device.upper()}")
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Veriyi Tokenize Etme Fonksiyonu
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    print("Veriler tokenize ediliyor...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Model Tanımlama (4 sınıf için)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
    
    # Metrik Hesaplama Fonksiyonu
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, predictions)}
    
    # Eğitim Argümanları
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1, # Test amaçlı 1 epoch (isteğe göre artırılabilir)
        weight_decay=00.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir='./logs',
        logging_steps=100,
    )
    
    # Trainer Kurulumu
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range=10000), # Hızlı eğitim için alt küme
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )
    
    print("DistilBERT ince ayar (fine-tuning) işlemi başlıyor...")
    trainer.train()
    
    print("\nModel değerlendiriliyor...")
    eval_results = trainer.evaluate()
    print(f"DistilBERT Test Doğruluğu: {eval_results['eval_accuracy']:.4f}")


if __name__ == "__main__":
    # Hangi modeli çalıştırmak istiyorsan aşağıdaki yorum satırını kaldırabilirsin:
    
    run_random_forest()
    # run_distilbert()
