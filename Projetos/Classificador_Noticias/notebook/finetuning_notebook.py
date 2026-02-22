"""
Fine-tuning BERT para Classificação de Notícias
Notebook: finetuning_bert_noticias.ipynb

Este arquivo contém o código completo para:
1. Carregar dataset AG News
2. Preparar dados
3. Fine-tuning de DistilBERT
4. Avaliar modelo
5. Salvar modelo treinado

Autor: Carlos Henrique Bamberg Marques
Módulo: EAI_05 - NLP com Transformers
"""

# ========================================
# CÉLULA 1: Imports e Setup
# ========================================

# Instalar dependências (executar uma vez)
# !pip install transformers datasets torch scikit-learn matplotlib seaborn

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

print("✅ Imports realizados com sucesso")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ========================================
# CÉLULA 2: Carregar Dataset AG News
# ========================================

print("\n📥 Carregando dataset AG News...")

# Carregar via Hugging Face Datasets
dataset = load_dataset('ag_news')

print(f"\n✅ Dataset carregado:")
print(f"  - Treino: {len(dataset['train'])} exemplos")
print(f"  - Teste:  {len(dataset['test'])} exemplos")

# Visualizar exemplos
print("\n📋 Exemplos do dataset:")
for i in range(3):
    exemplo = dataset['train'][i]
    label_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    print(f"\nExemplo {i+1}:")
    print(f"  Classe: {label_names[exemplo['label']]}")
    print(f"  Texto:  {exemplo['text'][:100]}...")

# Distribuição das classes
print("\n📊 Distribuição das classes (Treino):")
labels = [ex['label'] for ex in dataset['train']]
for i, name in enumerate(['World', 'Sports', 'Business', 'Sci/Tech']):
    count = labels.count(i)
    pct = count / len(labels) * 100
    print(f"  {name:12} - {count:6} ({pct:.1f}%)")

# ========================================
# CÉLULA 3: Preparar Tokenizer
# ========================================

print("\n🔧 Preparando tokenizer...")

MODEL_NAME = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    """Tokeniza os textos"""
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=512
    )

# Aplicar tokenização
print("⏳ Tokenizando dataset (pode levar alguns minutos)...")

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['text'],
    desc="Tokenizando"
)

print("\n✅ Tokenização completa")
print(f"Colunas: {tokenized_datasets['train'].column_names}")

# Exemplo tokenizado
print("\n📋 Exemplo tokenizado:")
exemplo = tokenized_datasets['train'][0]
print(f"  Input IDs shape: {len(exemplo['input_ids'])}")
print(f"  Attention mask:  {len(exemplo['attention_mask'])}")
print(f"  Label:           {exemplo['label']}")

# ========================================
# CÉLULA 4: Carregar Modelo Base
# ========================================

print("\n🤖 Carregando modelo DistilBERT base...")

modelo = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=4  # 4 classes
)

print("\n✅ Modelo carregado")
print(f"\nArquitetura:")
print(modelo)

# Contar parâmetros
total_params = sum(p.numel() for p in modelo.parameters())
trainable_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)

print(f"\n📊 Parâmetros:")
print(f"  Total:      {total_params:,}")
print(f"  Treináveis: {trainable_params:,}")

# ========================================
# CÉLULA 5: Definir Métricas
# ========================================

def compute_metrics(eval_pred):
    """
    Calcula métricas de avaliação
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Precision, Recall, F1 (macro average)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average='macro'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

print("✅ Função de métricas definida")

# ========================================
# CÉLULA 6: Training Arguments
# ========================================

print("\n⚙️ Configurando argumentos de treinamento...")

training_args = TrainingArguments(
    # Output
    output_dir='./results',
    
    # Epochs
    num_train_epochs=3,
    
    # Batch sizes
    per_device_train_batch_size=16,   # Ajustar se OOM
    per_device_eval_batch_size=64,
    
    # Learning rate
    learning_rate=2e-5,
    weight_decay=0.01,
    
    # Warmup
    warmup_steps=500,
    
    # Logging
    logging_dir='./logs',
    logging_steps=100,
    
    # Evaluation & Saving
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=2,
    
    # Best model
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    
    # Mixed precision (FP16) - se GPU suporta
    fp16=torch.cuda.is_available(),
    
    # Report
    report_to='none'  # Desabilitar W&B
)

print("✅ Argumentos configurados")
print(f"\nConfiguração:")
print(f"  Epochs:                {training_args.num_train_epochs}")
print(f"  Batch size (treino):   {training_args.per_device_train_batch_size}")
print(f"  Learning rate:         {training_args.learning_rate}")
print(f"  Warmup steps:          {training_args.warmup_steps}")
print(f"  FP16:                  {training_args.fp16}")

# ========================================
# CÉLULA 7: Criar Trainer
# ========================================

print("\n🏋️ Criando Trainer...")

trainer = Trainer(
    model=modelo,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("✅ Trainer criado")

# ========================================
# CÉLULA 8: TREINAR MODELO
# ========================================

print("\n" + "="*60)
print("🚀 INICIANDO FINE-TUNING")
print("="*60)
print("\n⏰ Tempo estimado: ~30 min (GPU) ou ~3 horas (CPU)")
print("💡 Dica: Acompanhe o progresso abaixo\n")

# TREINAR
train_result = trainer.train()

print("\n" + "="*60)
print("✅ TREINAMENTO CONCLUÍDO")
print("="*60)

# Métricas de treino
print(f"\n📊 Métricas finais de treino:")
print(f"  Loss:           {train_result.training_loss:.4f}")
print(f"  Tempo total:    {train_result.metrics['train_runtime']:.2f}s")
print(f"  Samples/sec:    {train_result.metrics['train_samples_per_second']:.2f}")

# ========================================
# CÉLULA 9: Avaliar no Conjunto de Teste
# ========================================

print("\n📊 Avaliando no conjunto de teste...")

eval_results = trainer.evaluate()

print("\n✅ Avaliação completa")
print(f"\n📈 Resultados no teste:")
print(f"  Accuracy:  {eval_results['eval_accuracy']:.4f} ({eval_results['eval_accuracy']*100:.2f}%)")
print(f"  Precision: {eval_results['eval_precision']:.4f}")
print(f"  Recall:    {eval_results['eval_recall']:.4f}")
print(f"  F1-Score:  {eval_results['eval_f1']:.4f}")

# ========================================
# CÉLULA 10: Predições e Confusion Matrix
# ========================================

print("\n🔮 Gerando predições para análise detalhada...")

# Predições
predictions_output = trainer.predict(tokenized_datasets['test'])
y_pred = np.argmax(predictions_output.predictions, axis=-1)
y_true = predictions_output.label_ids

# Classification Report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)

target_names = ['World', 'Sports', 'Business', 'Sci/Tech']
print(classification_report(y_true, y_pred, target_names=target_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Plotar
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=target_names,
    yticklabels=target_names,
    cbar_kws={'label': 'Contagem'}
)
plt.title('Matriz de Confusão - AG News Classification', fontsize=16, pad=20)
plt.ylabel('Classe Real', fontsize=12)
plt.xlabel('Classe Predita', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n💾 Matriz de confusão salva em 'confusion_matrix.png'")

# Análise de erros
print("\n📊 Análise de erros por classe:")
for i, name in enumerate(target_names):
    correct = cm[i, i]
    total = cm[i].sum()
    accuracy = correct / total * 100
    errors = total - correct
    print(f"  {name:12} - {correct}/{total} corretos ({accuracy:.1f}%) - {errors} erros")

# ========================================
# CÉLULA 11: Curva de Aprendizado
# ========================================

print("\n📈 Plotando curva de aprendizado...")

# Extrair histórico
history = trainer.state.log_history

# Separar treino e validação
train_loss = [x['loss'] for x in history if 'loss' in x]
eval_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]
eval_accuracy = [x['eval_accuracy'] for x in history if 'eval_accuracy' in x]

# Plotar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss
steps_train = range(len(train_loss))
steps_eval = np.linspace(0, len(train_loss), len(eval_loss))

ax1.plot(steps_train, train_loss, label='Training Loss', alpha=0.7)
ax1.plot(steps_eval, eval_loss, label='Validation Loss', marker='o')
ax1.set_xlabel('Steps', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training & Validation Loss', fontsize=14)
ax1.legend()
ax1.grid(alpha=0.3)

# Accuracy
epochs = range(1, len(eval_accuracy) + 1)
ax2.plot(epochs, eval_accuracy, marker='o', linewidth=2, markersize=8)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Validation Accuracy', fontsize=14)
ax2.set_ylim([0.8, 1.0])
ax2.grid(alpha=0.3)

# Anotar valores
for i, acc in enumerate(eval_accuracy):
    ax2.annotate(f'{acc:.3f}', 
                xy=(i+1, acc), 
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=10)

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("💾 Curvas salvas em 'learning_curves.png'")

# ========================================
# CÉLULA 12: Salvar Modelo Fine-tunado
# ========================================

print("\n💾 Salvando modelo fine-tunado...")

# Criar diretório se não existir
import os
save_dir = './models/bert_news_classifier'
os.makedirs(save_dir, exist_ok=True)

# Salvar modelo e tokenizer
modelo.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"\n✅ Modelo salvo em '{save_dir}'")
print("\nArquivos salvos:")
for file in os.listdir(save_dir):
    size = os.path.getsize(os.path.join(save_dir, file)) / (1024*1024)
    print(f"  - {file:30} ({size:.2f} MB)")

# ========================================
# CÉLULA 13: Testar Modelo Salvo
# ========================================

print("\n🧪 Testando modelo salvo...")

# Carregar modelo
modelo_carregado = AutoModelForSequenceClassification.from_pretrained(save_dir)
tokenizer_carregado = AutoTokenizer.from_pretrained(save_dir)

modelo_carregado.eval()

# Função de teste
def classificar(texto):
    """Classifica um texto"""
    inputs = tokenizer_carregado(
        texto,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        padding=True
    )
    
    with torch.no_grad():
        outputs = modelo_carregado(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred = torch.argmax(probs)
    
    classe = target_names[pred]
    confianca = probs[pred].item()
    
    return classe, confianca, probs.numpy()

# Exemplos de teste
exemplos = {
    'World': "UN Security Council discusses Middle East crisis",
    'Sports': "Lakers win championship in overtime thriller",
    'Business': "Stock market reaches new all-time high",
    'Sci/Tech': "New AI breakthrough in quantum computing"
}

print("\n🔍 Testando com exemplos:")
print("="*80)

for categoria_esperada, texto in exemplos.items():
    classe, conf, probs = classificar(texto)
    
    print(f"\n📰 Texto: {texto}")
    print(f"   Esperado: {categoria_esperada}")
    print(f"   Predito:  {classe} ({'✅' if classe == categoria_esperada else '❌'})")
    print(f"   Confiança: {conf:.2%}")
    print(f"\n   Probabilidades:")
    for i, name in enumerate(target_names):
        bar = '█' * int(probs[i] * 50)
        print(f"     {name:12} {probs[i]:.2%} {bar}")

print("\n" + "="*80)

# ========================================
# CÉLULA 14: Resumo Final
# ========================================

print("\n" + "="*80)
print("🎉 FINE-TUNING CONCLUÍDO COM SUCESSO!")
print("="*80)

print(f"""
📊 Resumo do Treinamento:
   - Modelo:               DistilBERT
   - Dataset:              AG News (120k treino, 7.6k teste)
   - Classes:              4 (World, Sports, Business, Sci/Tech)
   - Epochs:               {training_args.num_train_epochs}
   - Batch size:           {training_args.per_device_train_batch_size}
   - Learning rate:        {training_args.learning_rate}
   
📈 Resultados:
   - Accuracy final:       {eval_results['eval_accuracy']:.2%}
   - F1-Score:             {eval_results['eval_f1']:.4f}
   - Melhor modelo salvo:  {save_dir}
   
📁 Arquivos gerados:
   - Modelo fine-tunado:   {save_dir}/
   - Confusion matrix:     confusion_matrix.png
   - Learning curves:      learning_curves.png
   
🚀 Próximos passos:
   1. Execute app.py para testar no navegador
   2. Use o modelo via API REST
   3. Deploy em produção (Heroku, AWS, etc.)
   
💡 Comparação com EAI_04:
   - TF-IDF + SVM:         ~87% accuracy
   - BERT Fine-tuned:      ~{eval_results['eval_accuracy']*100:.1f}% accuracy
   - Ganho:                +{(eval_results['eval_accuracy']*100 - 87):.1f} pontos
""")

print("="*80)
print("✅ Notebook executado com sucesso!")
print("="*80)

# FIM DO NOTEBOOK
