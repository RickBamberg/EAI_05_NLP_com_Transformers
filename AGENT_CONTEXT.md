# AGENT_CONTEXT.md - EAI_05 NLP com Transformers

> **Propósito**: Contexto técnico completo do módulo EAI_05  
> **Última atualização**: Janeiro 2026  
> **Tipo**: Módulo educacional com 5 notebooks progressivos

## RESUMO EXECUTIVO

**Objetivo**: Ensinar arquitetura Transformer e Hugging Face  
**Notebooks**: 5 notebooks (teoria → prática)  
**Técnicas**: Self-Attention, Positional Encoding, Transfer Learning  
**Modelos**: DistilBERT, DialoGPT  
**Biblioteca**: Hugging Face Transformers  
**Diferencial**: Do zero (matemática) até aplicações (chatbot)

---

## ARQUITETURA TRANSFORMER - VISÃO TÉCNICA

### Por Que Transformers Revolucionaram NLP

**Problema com RNNs/LSTMs**:
```python
# Processamento sequencial (lento)
h_t = f(h_{t-1}, x_t)

# Dependências de longo prazo difíceis
"O gato que estava no jardim comeu."
# "gato" → "comeu" precisa propagar por 6 steps
```

**Solução: Self-Attention**:
```python
# Paralelização total
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

# Todas as palavras se conectam diretamente
"O gato que estava no jardim comeu."
# "gato" ← → "comeu" em 1 step!
```

### Arquitetura Completa

```
Input Tokens
    ↓
Embedding Layer (vocab_size → d_model)
    ↓
Positional Encoding (adiciona posição)
    ↓
Encoder Stack (N=6 camadas)
│   ├─ Multi-Head Self-Attention
│   ├─ Add & Norm
│   ├─ Feed-Forward Network
│   └─ Add & Norm
    ↓
Decoder Stack (N=6 camadas)
│   ├─ Masked Multi-Head Self-Attention
│   ├─ Add & Norm
│   ├─ Cross-Attention (com Encoder)
│   ├─ Add & Norm
│   ├─ Feed-Forward Network
│   └─ Add & Norm
    ↓
Linear Layer + Softmax
    ↓
Output Probabilities
```

---

## NOTEBOOK 1: transformers_basico.ipynb

### Objetivo Pedagógico
Entender fundamentos matemáticos antes de usar bibliotecas.

### Self-Attention - Matemática Detalhada

```python
import numpy as np

# Input: 3 palavras, 4 dimensões cada
X = np.array([
    [1, 0, 1, 0],  # palavra 1
    [0, 2, 0, 2],  # palavra 2
    [1, 1, 1, 1]   # palavra 3
])
# Shape: (3, 4)

# Matrizes de peso (inicializadas aleatoriamente)
d_model = 4
W_Q = np.random.rand(d_model, d_model)
W_K = np.random.rand(d_model, d_model)
W_V = np.random.rand(d_model, d_model)

# 1. Projeções lineares
Q = X @ W_Q  # Queries:  (3, 4)
K = X @ W_K  # Keys:     (3, 4)
V = X @ W_V  # Values:   (3, 4)

# 2. Calcular scores
scores = Q @ K.T  # (3, 3)
# scores[i,j] = quão relevante é palavra j para palavra i

# 3. Escalar (evita gradientes instáveis)
d_k = K.shape[-1]
scores_scaled = scores / np.sqrt(d_k)

# 4. Softmax (normalizar em probabilidades)
import scipy.special
attention_weights = scipy.special.softmax(scores_scaled, axis=-1)
# Cada linha soma 1

# 5. Aplicar atenção aos valores
output = attention_weights @ V  # (3, 4)
```

**Interpretação**:
```python
# attention_weights[i,j] = quanto palavra i "atende" palavra j
#
# Exemplo: attention_weights
# [[0.35, 0.33, 0.32],  ← palavra 1 atende todas igualmente
#  [0.29, 0.41, 0.30],  ← palavra 2 atende mais ela mesma
#  [0.33, 0.32, 0.35]]  ← palavra 3 atende todas igualmente
```

### Multi-Head Attention

**Por que múltiplas "cabeças"?**
```python
# Uma cabeça pode focar em:
# - Sintaxe ("verbo" → "sujeito")
# Outra cabeça:
# - Semântica ("gato" → "animal")
# Outra:
# - Posição ("próximo" → "palavra seguinte")

# Implementação
n_heads = 8
d_k = d_model // n_heads  # 64 / 8 = 8 dimensões por cabeça

# Para cada cabeça h:
for h in range(n_heads):
    Q_h = Q @ W_Q_h  # (seq_len, d_k)
    K_h = K @ W_K_h
    V_h = V @ W_V_h
    
    attention_h = Attention(Q_h, K_h, V_h)

# Concatenar todas as cabeças
output = Concat(attention_1, ..., attention_8) @ W_O
```

### Positional Encoding - Implementação

```python
def positional_encoding(seq_len, d_model):
    """
    Gera matriz de codificação posicional
    
    Args:
        seq_len: comprimento da sequência
        d_model: dimensões do modelo (deve ser par)
    
    Returns:
        PE: matriz (seq_len, d_model)
    """
    PE = np.zeros((seq_len, d_model))
    
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            # Seno para dimensões pares
            PE[pos, i] = np.sin(pos / (10000 ** (2*i / d_model)))
            
            # Cosseno para dimensões ímpares
            if i+1 < d_model:
                PE[pos, i+1] = np.cos(pos / (10000 ** (2*i / d_model)))
    
    return PE

# Uso
PE = positional_encoding(seq_len=100, d_model=512)
# Shape: (100, 512)

# Adicionar aos embeddings
embeddings_with_position = word_embeddings + PE
```

**Por que essa fórmula específica?**
```python
# Propriedades importantes:
# 1. Valores limitados [-1, 1]
# 2. Único para cada posição
# 3. Relações relativas: PE[pos+k] = f(PE[pos])
# 4. Generaliza para sequências maiores que as de treino
```

---

## NOTEBOOK 2: tokenizacao_transformers.ipynb

### Objetivo Pedagógico
Dominar tokenização - passo crítico e muitas vezes negligenciado.

### Tokenização: Subword vs Word vs Character

```python
# Character-level (não usado em Transformers)
"hello" → ['h', 'e', 'l', 'l', 'o']
# Problema: sequências muito longas

# Word-level
"hello world" → ['hello', 'world']
# Problema: vocabulário gigante, OOV words

# Subword (usado em BERT, GPT) ✓
"unhappiness" → ['un', '##happiness']
# Vantagens:
# - Vocabulário limitado (~30k tokens)
# - Sem OOV
# - Captura morfologia
```

### WordPiece (usado em BERT)

```python
# Algoritmo simplificado
vocab = ['[CLS]', '[SEP]', '[PAD]', '[UNK]', 'a', 'b', ...]

def tokenize_wordpiece(word, vocab):
    tokens = []
    start = 0
    
    while start < len(word):
        end = len(word)
        found = False
        
        # Tentar maior substring possível
        while start < end:
            substr = word[start:end]
            if start > 0:
                substr = "##" + substr  # Continuação
            
            if substr in vocab:
                tokens.append(substr)
                found = True
                break
            
            end -= 1
        
        if not found:
            tokens.append('[UNK]')
            break
        
        start = end
    
    return tokens

# Exemplo
tokenize_wordpiece("unhappiness", vocab)
# ['un', '##happiness'] ou ['un', '##happ', '##iness']
```

### Padding e Attention Mask

```python
# Problema: batches precisam de mesmo tamanho
frases = [
    "Hello world",           # 3 tokens (+ [CLS], [SEP])
    "I love AI very much"    # 6 tokens
]

# Solução: Padding
tokenizer(frases, padding=True)
# Resultado:
# [
#   [101, 7592, 2088, 102, 0, 0, 0],  # "Hello world" + 3 pads
#   [101, 1045, 2293, 9932, 2200, 2172, 102]  # "I love AI very much"
# ]

# Attention Mask (importante!)
# [
#   [1, 1, 1, 1, 0, 0, 0],  # 1=token real, 0=padding
#   [1, 1, 1, 1, 1, 1, 1]
# ]

# Uso no modelo
attention_scores = Q @ K.T
attention_scores = attention_scores.masked_fill(mask==0, -1e9)
# Padding recebe score -∞ → após softmax ≈ 0
```

### Truncation

```python
# Problema: Modelos têm tamanho máximo
# BERT: 512 tokens
# GPT-2: 1024 tokens

frase_longa = "..." * 1000  # 3000 caracteres

# Truncation
tokenizer(frase_longa, truncation=True, max_length=512)
# Mantém primeiros 512 tokens

# Estratégias de truncation
truncation='longest_first'  # Corta a mais longa primeiro (útil para pares)
truncation='only_first'     # Corta apenas primeira frase
truncation='only_second'    # Corta apenas segunda
```

---

## NOTEBOOK 3: modelo_transformers.ipynb

### Objetivo Pedagógico
Entender output de modelos Transformer e como usar embeddings.

### AutoModel vs AutoModelForX

```python
from transformers import AutoModel, AutoModelForSequenceClassification

# AutoModel: modelo "nu" (apenas embeddings)
modelo_base = AutoModel.from_pretrained('bert-base-uncased')
# Output: embeddings (sem cabeça de classificação)

# AutoModelForSequenceClassification: com cabeça
modelo_cls = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)
# Output: logits para 2 classes
```

### Estrutura de Output

```python
# Forward pass
outputs = modelo(**inputs)

# Outputs tem:
outputs.last_hidden_state  # (batch, seq_len, hidden_size)
outputs.pooler_output      # (batch, hidden_size) - BERT only
outputs.hidden_states      # Todas as camadas (se output_hidden_states=True)
outputs.attentions         # Pesos de atenção (se output_attentions=True)
```

### Extrair Embeddings para Classificação

```python
import torch
import torch.nn as nn

class ClassificadorCustom(nn.Module):
    def __init__(self, modelo_base, num_classes):
        super().__init__()
        self.bert = modelo_base
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attention_mask):
        # 1. Passar pelo BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 2. Pegar [CLS] token (primeira posição)
        cls_output = outputs.last_hidden_state[:, 0, :]
        # Shape: (batch, 768)
        
        # 3. Dropout
        cls_output = self.dropout(cls_output)
        
        # 4. Classificador
        logits = self.classifier(cls_output)
        # Shape: (batch, num_classes)
        
        return logits
```

### Embeddings para Similaridade

```python
def get_sentence_embedding(texto, tokenizer, modelo):
    """
    Extrai embedding de uma frase
    """
    inputs = tokenizer(texto, return_tensors='pt')
    
    with torch.no_grad():
        outputs = modelo(**inputs)
    
    # Mean pooling (média de todos os tokens)
    embeddings = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    
    # Expandir attention_mask para broadcasting
    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    
    # Soma ponderada
    sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    
    return sum_embeddings / sum_mask

# Uso
emb1 = get_sentence_embedding("I love AI", tokenizer, modelo)
emb2 = get_sentence_embedding("AI is amazing", tokenizer, modelo)

# Similaridade de cosseno
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(emb1, emb2)
# ~0.85 (alta similaridade)
```

---

## NOTEBOOK 4: classificacao_sentimentos_transformers.ipynb

### Objetivo Pedagógico
Usar modelos pré-treinados para tarefas práticas (transfer learning).

### Pipeline - Abstração de Alto Nível

```python
from transformers import pipeline

# Pipeline encapsula:
# 1. Tokenizer
# 2. Modelo
# 3. Post-processing

classifier = pipeline('sentiment-analysis')

# Internamente faz:
# 1. tokenizer(texto) → input_ids, attention_mask
# 2. modelo(input_ids, attention_mask) → logits
# 3. softmax(logits) → probabilidades
# 4. argmax → label
```

### Modelos Disponíveis

```python
# Sentiment Analysis
pipeline('sentiment-analysis')
# Modelo default: distilbert-base-uncased-finetuned-sst-2-english

# Named Entity Recognition
pipeline('ner')
# Modelo default: dbmdz/bert-large-cased-finetuned-conll03-english

# Question Answering
pipeline('question-answering')
# Modelo default: distilbert-base-cased-distilled-squad

# Text Generation
pipeline('text-generation')
# Modelo default: gpt2

# Translation
pipeline('translation_en_to_fr')
# Modelo default: t5-base

# Summarization
pipeline('summarization')
# Modelo default: sshleifer/distilbart-cnn-12-6
```

### Especificar Modelo Customizado

```python
# Carregar modelo específico
classifier = pipeline(
    'sentiment-analysis',
    model='nlptown/bert-base-multilingual-uncased-sentiment'
)

# Este modelo tem 5 classes (1-5 estrelas)
result = classifier("This is amazing!")
# [{'label': '5 stars', 'score': 0.89}]
```

### Batch Processing (Eficiente)

```python
# Ruim: Loop (lento)
for frase in frases:
    result = classifier(frase)

# Bom: Batch (rápido)
resultados = classifier(frases)  # Lista de frases

# Melhor: Batch com tamanho definido
resultados = classifier(frases, batch_size=32)
# Processa 32 frases por vez (otimizado para GPU)
```

---

## NOTEBOOK 5: chatbot_transformers.ipynb

### Objetivo Pedagógico
Geração de texto com modelos causais (GPT-style).

### Causal Language Model vs Masked Language Model

```python
# Masked LM (BERT):
# Input:  "O [MASK] está na mesa"
# Output: "gato" (preenche máscara)
# Uso: Classificação, QA

# Causal LM (GPT):
# Input:  "O gato está"
# Output: "na mesa" (continua texto)
# Uso: Geração de texto, chatbot
```

### DialoGPT - Arquitetura

```python
# Baseado em GPT-2
# Treinado em ~147M conversas do Reddit
# Input: Histórico de conversação
# Output: Próxima resposta

# Exemplo de treino:
# User:    "What's your favorite color?"
# Bot:     "Blue, what's yours?"
# User:    "Red."
# Bot:     "Nice choice!"

# Modelo aprende:
# P(resposta | pergunta + histórico)
```

### Generate - Parâmetros Importantes

```python
resposta_ids = modelo.generate(
    input_ids,
    
    # Tamanho
    max_length=1000,           # Comprimento máximo total
    min_length=10,             # Comprimento mínimo
    
    # Sampling
    do_sample=True,            # Amostragem estocástica (vs greedy)
    temperature=0.7,           # Controla "criatividade"
                              # <1: Mais conservador
                              # >1: Mais aleatório
    
    # Top-k sampling
    top_k=50,                  # Considera apenas top-50 tokens
    
    # Top-p (nucleus) sampling
    top_p=0.95,                # Considera tokens até 95% de prob acumulada
    
    # Repetição
    repetition_penalty=1.2,    # Penaliza tokens repetidos
    no_repeat_ngram_size=3,    # Não repete 3-gramas
    
    # Tokens especiais
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    
    # Controle
    num_return_sequences=1,    # Quantas sequências gerar
)
```

### Temperatura - Efeito Prático

```python
# Temperatura = 0.1 (conservador)
# "I like cats"
# → "I like cats because they are cute and fluffy."
# (Previsível, coerente)

# Temperatura = 1.0 (balanceado)
# "I like cats"
# → "I like cats, especially the orange ones!"
# (Criativo mas coerente)

# Temperatura = 2.0 (muito criativo)
# "I like cats"
# → "I like cats purple dancing quantum elephants."
# (Incoerente, aleatório demais)
```

### Gerenciar Histórico de Conversação

```python
def chat_loop():
    chat_history_ids = None
    
    while True:
        # 1. Receber input
        user_input = input("Você: ")
        if user_input.lower() == 'sair':
            break
        
        # 2. Encode input
        new_input_ids = tokenizer.encode(
            user_input + tokenizer.eos_token,
            return_tensors='pt'
        )
        
        # 3. Concatenar com histórico
        if chat_history_ids is not None:
            input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
        else:
            input_ids = new_input_ids
        
        # 4. Gerar resposta
        chat_history_ids = modelo.generate(
            input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # 5. Decode apenas a nova resposta
        resposta = tokenizer.decode(
            chat_history_ids[:, input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )
        
        print(f"Bot: {resposta}")
```

**Importante**: Histórico cresce indefinidamente
```python
# Solução: Sliding window
MAX_HISTORY = 500  # tokens

if chat_history_ids.shape[-1] > MAX_HISTORY:
    chat_history_ids = chat_history_ids[:, -MAX_HISTORY:]
```

---

## COMPARAÇÃO TÉCNICA: MODELOS

### BERT vs GPT vs T5

| Aspecto | BERT | GPT | T5 |
|---------|------|-----|-----|
| **Arquitetura** | Encoder | Decoder | Encoder-Decoder |
| **Treinamento** | Masked LM | Causal LM | Span Corruption |
| **Bidirecional** | ✅ Sim | ❌ Não | ✅ Sim (encoder) |
| **Geração** | ❌ Ruim | ✅ Excelente | ✅ Boa |
| **Classificação** | ✅ Excelente | ⚠️ OK | ✅ Boa |
| **QA** | ✅ Excelente | ⚠️ OK | ✅ Excelente |
| **Resumo** | ❌ Não | ⚠️ OK | ✅ Excelente |
| **Tradução** | ❌ Não | ❌ Não | ✅ Excelente |

### DistilBERT - Conhecimento Técnico

**Destilação de Conhecimento**:
```python
# Teacher (BERT original)
logits_teacher = BERT(input)

# Student (DistilBERT)
logits_student = DistilBERT(input)

# Loss
loss_distillation = KL_divergence(
    softmax(logits_student / T),
    softmax(logits_teacher / T)
)

# T = temperatura (smooth distributions)
```

**Comparação**:
```
BERT-base:
- 12 camadas
- 768 hidden size
- 110M parâmetros
- ~100% performance

DistilBERT:
- 6 camadas
- 768 hidden size
- 66M parâmetros (-40%)
- ~97% performance
- ~2x mais rápido
```

---

## TROUBLESHOOTING COMUM

### Problema 1: Out of Memory (OOM)

```python
# Causas:
# - Batch size muito grande
# - Sequência muito longa
# - Modelo muito grande para GPU

# Soluções:
# 1. Reduzir batch_size
batch_size = 8  # em vez de 32

# 2. Gradient accumulation
accumulation_steps = 4
# Efetivo batch_size = 8 * 4 = 32

# 3. Mixed precision (FP16)
from torch.cuda.amp import autocast
with autocast():
    outputs = modelo(**inputs)

# 4. Usar modelo menor
modelo = AutoModel.from_pretrained('distilbert-base-uncased')  # em vez de bert-large
```

### Problema 2: Geração Repetitiva

```python
# Solução 1: Repetition penalty
modelo.generate(
    input_ids,
    repetition_penalty=1.2  # Penaliza repetição
)

# Solução 2: No repeat n-grams
modelo.generate(
    input_ids,
    no_repeat_ngram_size=3  # Não repete 3-gramas
)

# Solução 3: Aumentar temperatura
modelo.generate(
    input_ids,
    temperature=0.8  # Mais variado
)
```

### Problema 3: Tokenização Lenta

```python
# Ruim: Loop
for texto in textos:
    tokens = tokenizer(texto)

# Bom: Batch
tokens = tokenizer(textos)  # Lista de textos

# Melhor: Fast tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    'bert-base-uncased',
    use_fast=True  # Usa Rust backend (10x+ rápido)
)
```

---

## MÉTRICAS DE PERFORMANCE

### Classificação

```python
from sklearn.metrics import classification_report

y_true = [0, 1, 0, 1, 1]
y_pred = [0, 1, 0, 0, 1]

print(classification_report(y_true, y_pred))
```

### Geração de Texto

```
BLEU: Overlap de n-gramas com referência
ROUGE: Recall de n-gramas
Perplexity: exp(cross_entropy_loss)
```

---

## TAGS DE BUSCA

`#transformers` `#self-attention` `#bert` `#gpt` `#huggingface` `#nlp-moderno` `#distilbert` `#dialogpt` `#transfer-learning` `#positional-encoding` `#tokenization` `#pytorch`

---

**Versão**: 1.0  
**Compatibilidade**: Python 3.7+, transformers 4.30+, torch 2.0+  
**Uso recomendado**: Aprendizado de Transformers, aplicações de NLP moderno
