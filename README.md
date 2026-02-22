# EAI_05 - NLP com Transformers

Módulo completo de **NLP Moderno** usando a arquitetura Transformer e a biblioteca Hugging Face. Aprenda desde os fundamentos (Self-Attention, Positional Encoding) até aplicações práticas (classificação, chatbots).

---

## 🎯 Objetivo do Módulo

Dominar Transformers - a arquitetura que revolucionou NLP:
- ✅ Entender Self-Attention e como supera RNNs/LSTMs
- ✅ Usar modelos pré-treinados (BERT, GPT, DistilBERT)
- ✅ Aplicar Hugging Face em tarefas reais
- ✅ Construir aplicações de NLP moderno

**Por que Transformers?**
- Paralelização eficiente (vs RNNs sequenciais)
- Captura dependências de longo alcance
- Transfer Learning (modelos pré-treinados)
- Estado-da-arte em praticamente todas as tarefas de NLP

---

## 📂 Estrutura do Módulo

```
EAI_05_NLP_com_Transformers/
├── README.md (este arquivo)
├── AGENT_CONTEXT.md
│
└── Notebooks/
    ├── transformers_basico.ipynb
    ├── tokenizacao_transformers.ipynb
    ├── modelo_transformers.ipynb
    ├── classificacao_sentimentos_transformers.ipynb
    └── chatbot_transformers.ipynb
```

**Total**: 5 notebooks progressivos (teoria → prática)

---

## 🗺️ Jornada de Aprendizado

### Progressão Recomendada

```
Semana 1: Fundamentos
├── Dia 1-2: transformers_basico.ipynb
│   └─ Conceitos, Self-Attention, Arquitetura
│
├── Dia 3-4: tokenizacao_transformers.ipynb
│   └─ AutoTokenizer, padding, truncation
│
└── Dia 5-6: modelo_transformers.ipynb
    └─ Carregar modelo, extrair embeddings

Semana 2: Aplicações
├── Dia 1-3: classificacao_sentimentos_transformers.ipynb
│   └─ Pipeline, análise de sentimento
│
└── Dia 4-6: chatbot_transformers.ipynb
    └─ DialoGPT, conversação
```

**Tempo estimado**: 2 semanas (dedicação parcial)

---

## 📚 Conteúdo Detalhado

### 1️⃣ transformers_basico.ipynb

**Objetivo**: Fundamentos da arquitetura Transformer

#### Conceitos Abordados:
- ❌ **Limitações de RNNs/LSTMs**
  - Dependências de longo prazo difíceis
  - Processamento sequencial (lento)
  - Difícil paralelizar

- ✅ **Arquitetura Transformer**
  - Encoder + Decoder
  - Self-Attention como core
  - Paralelização total

#### Self-Attention (Implementação Simplificada):
```python
import numpy as np

# 3 palavras, 4 dimensões
X = np.array([[1,0,1,0], [0,2,0,2], [1,1,1,1]])

# Projeções Q, K, V
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

# Scores de atenção
scores = Q @ K.T / np.sqrt(d_k)
weights = softmax(scores)

# Output
output = weights @ V
```

#### Positional Encoding:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

**Por quê?** Transformers não têm noção de ordem embutida (como RNNs).

#### Modelos Baseados em Transformer:

| Modelo | Tipo | Uso Típico |
|--------|------|------------|
| **BERT** | Encoder (bidirecional) | Classificação, QA |
| **GPT** | Decoder (autoregressivo) | Geração de texto |
| **T5/BART** | Encoder-Decoder | Tradução, resumo |

---

### 2️⃣ tokenizacao_transformers.ipynb

**Objetivo**: Dominar tokenização com Hugging Face

#### AutoTokenizer:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenizar
frase = "Transformers são poderosos!"
tokens = tokenizer.tokenize(frase)
# ['transformers', 'sao', 'pod', '##eros', '##os', '!']

ids = tokenizer.convert_tokens_to_ids(tokens)
# [19081, 7509, 17491, 27360, 2891, 999]
```

#### Padding & Truncation:
```python
frases = [
    "Frase curta",
    "Frase muito mais longa que a primeira"
]

encodings = tokenizer(
    frases,
    padding=True,        # Iguala tamanhos
    truncation=True,     # Corta se >max_length
    return_tensors="pt"  # PyTorch tensors
)

# encodings['input_ids']:     IDs dos tokens
# encodings['attention_mask']: Máscaras (1=real, 0=padding)
```

#### Encode & Decode:
```python
# Texto → IDs
ids = tokenizer.encode("Isso é incrível!", add_special_tokens=True)
# [101, 26354, 2080, 1041, 4297, 3089, 15985, 999, 102]
#  [CLS]  isso    e     ...                  !    [SEP]

# IDs → Texto
texto = tokenizer.decode(ids)
# "[CLS] isso e incrivel! [SEP]"
```

**Tokens Especiais**:
- `[CLS]`: Início da sequência (usado para classificação)
- `[SEP]`: Separador de frases
- `[PAD]`: Padding
- `[MASK]`: Máscara (para BERT)

---

### 3️⃣ modelo_transformers.ipynb

**Objetivo**: Usar modelo pré-treinado e extrair embeddings

#### Carregar Modelo:
```python
from transformers import AutoTokenizer, AutoModel
import torch

modelo_nome = 'distilbert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(modelo_nome)
modelo = AutoModel.from_pretrained(modelo_nome)

modelo.eval()  # Modo de inferência
```

**DistilBERT**:
- Versão destilada do BERT (menor, mais rápida)
- 6 camadas (vs 12 do BERT)
- ~97% da performance, ~40% menor

#### Extrair Embeddings:
```python
frase = "Transformers are powerful models for NLP tasks."
tokens = tokenizer(frase, return_tensors='pt')

with torch.no_grad():
    saida = modelo(**tokens)

# saida.last_hidden_state: shape (1, 11, 768)
#   1   = batch size
#   11  = tokens (incluindo [CLS], [SEP])
#   768 = dimensões do embedding

# Embedding do token [CLS] (representação da frase inteira)
vetor_cls = saida.last_hidden_state[0][0]  # shape (768,)
```

**Aplicações dos Embeddings**:
- Classificação: Use [CLS] → Dense Layer → Classes
- Similaridade: Compare vetores [CLS] de duas frases
- Features: Use como input para outro modelo

---

### 4️⃣ classificacao_sentimentos_transformers.ipynb

**Objetivo**: Análise de sentimento com pipeline

#### Pipeline Sentiment-Analysis:
```python
from transformers import pipeline

analisador = pipeline("sentiment-analysis")

frase = "I love this product!"
resultado = analisador(frase)

# Output:
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

#### Modelo Usado:
`distilbert-base-uncased-finetuned-sst-2-english`
- Treinado no Stanford Sentiment Treebank (SST-2)
- 2 classes: POSITIVE, NEGATIVE
- Accuracy: ~92%

#### Análise em Lote:
```python
frases = [
    "This is amazing!",
    "I hate this product.",
    "It's okay, nothing special."
]

resultados = analisador(frases)

for frase, res in zip(frases, resultados):
    print(f"{frase:30} → {res['label']:8} ({res['score']:.2%})")

# Output:
# This is amazing!             → POSITIVE (99.98%)
# I hate this product.         → NEGATIVE (99.95%)
# It's okay, nothing special.  → POSITIVE (52.31%)
```

#### Visualização:
```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(resultados)
df['frase'] = frases

# Gráfico de barras
plt.barh(df['frase'], df['score'], color=['green' if l=='POSITIVE' else 'red' for l in df['label']])
plt.xlabel('Confiança')
plt.title('Análise de Sentimento')
plt.show()
```

---

### 5️⃣ chatbot_transformers.ipynb

**Objetivo**: Criar chatbot conversacional com DialoGPT

#### DialoGPT:
Modelo da Microsoft treinado em ~147M conversas do Reddit
- Baseado em GPT-2
- Especializado em diálogo
- Versões: small, medium, large

#### Implementação:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

modelo_nome = "microsoft/DialoGPT-medium"

tokenizer = AutoTokenizer.from_pretrained(modelo_nome)
modelo = AutoModelForCausalLM.from_pretrained(modelo_nome)

def responder(pergunta, chat_historia_ids=None):
    # Encode pergunta
    nova_entrada_ids = tokenizer.encode(
        pergunta + tokenizer.eos_token,
        return_tensors="pt"
    )
    
    # Concatenar com histórico
    input_ids = torch.cat([chat_historia_ids, nova_entrada_ids], dim=-1) \
                if chat_historia_ids is not None else nova_entrada_ids
    
    # Gerar resposta
    resposta_ids = modelo.generate(
        input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    
    # Decode apenas a nova resposta
    resposta = tokenizer.decode(
        resposta_ids[:, input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )
    
    return resposta, resposta_ids
```

#### Loop Interativo:
```python
chat_historia_ids = None

print("Chatbot: Olá! Como posso ajudar?")
while True:
    entrada = input("Você: ")
    if entrada.lower() == "sair":
        break
    
    resposta, chat_historia_ids = responder(entrada, chat_historia_ids)
    print(f"Chatbot: {resposta}")
```

#### Limitações do DialoGPT:
- ⚠️ Treinado em conversas gerais (Reddit)
- ⚠️ Respostas podem ser vagas ou incoerentes
- ⚠️ Não tem conhecimento específico de domínio
- ⚠️ Pode gerar conteúdo inapropriado

#### Modelos Alternativos (Melhores):
| Modelo | Tamanho | Características |
|--------|---------|-----------------|
| **LLaMA 2 Chat** | 7B-70B | Conversação útil e segura |
| **Mistral Instruct** | 7B | Leve, rápido, direto |
| **Gemma** | 2B-7B | Alta qualidade, Google |

---

## 📊 Comparação: NLP Clássico vs Transformers

| Aspecto | NLP Clássico (EAI_04) | Transformers (EAI_05) |
|---------|----------------------|----------------------|
| **Representação** | TF-IDF (esparso) | Embeddings (denso) |
| **Contexto** | Bag of Words (sem ordem) | Self-Attention (captura relações) |
| **Modelo** | SVM, Naive Bayes | BERT, GPT |
| **Treino** | Minutos (CPU) | Horas (GPU) |
| **Tamanho** | <10 MB | 200 MB - 10 GB |
| **Accuracy** | 85-92% | 93-98% |
| **Quando usar** | Baseline rápido | Estado-da-arte |

---

## 💻 Instalação

### Requisitos
```
Python 3.7+
16GB RAM (recomendado)
GPU (opcional, mas acelera)
```

### Setup
```bash
# Criar ambiente
conda create -n transformers_env python=3.9
conda activate transformers_env

# Instalar dependências
pip install -r requirements.txt
```

### requirements.txt
```txt
transformers>=4.30.0
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
```

### Executar Notebooks
```bash
jupyter notebook
# ou
jupyter lab
```

---

## 🎯 Checklist de Conclusão

### Fundamentos
- [ ] Entendo Self-Attention matematicamente
- [ ] Sei a diferença entre Encoder e Decoder
- [ ] Conheço Positional Encoding
- [ ] Sei quando usar BERT vs GPT

### Prática
- [ ] Tokenizo texto com AutoTokenizer
- [ ] Extraio embeddings de modelo pré-treinado
- [ ] Uso pipeline para tarefas comuns
- [ ] Implementei chatbot simples

### Aplicação
- [ ] Classifico sentimento com Transformers
- [ ] Comparo com NLP clássico
- [ ] Conheço limitações e quando usar cada abordagem

---

## 🔮 Próximos Passos

Após completar EAI_05:

### EAI_06 - Fine-tuning de Transformers
- Adaptar BERT para domínio específico
- Treinar em datasets personalizados
- Técnicas de otimização (LoRA, QLoRA)

### EAI_07 - RAG e Aplicações Avançadas
- Retrieval Augmented Generation
- Vector databases (FAISS, Pinecone)
- Produção com LangChain

### Projetos Avançados
- Question Answering
- Named Entity Recognition (NER)
- Tradução automática
- Sumarização

---

## 📖 Recursos Complementares

### Papers Essenciais
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

### Cursos
- [Hugging Face Course](https://huggingface.co/learn/nlp-course)
- [Stanford CS224N: NLP with Deep Learning](http://web.stanford.edu/class/cs224n/)
- [Fast.ai: NLP](https://www.fast.ai/)

### Documentação
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [PyTorch Documentation](https://pytorch.org/docs/)

### Ferramentas
- [Hugging Face Spaces](https://huggingface.co/spaces) - Deploy rápido
- [Gradio](https://www.gradio.app/) - Interfaces para modelos
- [Streamlit](https://streamlit.io/) - Apps de ML

---

## 🤝 Contribuindo

Encontrou um erro? Tem uma sugestão?

1. Fork o repositório
2. Crie branch (`git checkout -b feature/melhoria`)
3. Commit mudanças
4. Push para branch
5. Abra Pull Request

---

## 📧 Contato

**Autor**: Carlos Henrique Bamberg Marques  
**Email**: rick.bamberg@gmail.com  
**GitHub**: [@RickBamberg](https://github.com/RickBamberg/)

---

## 📄 Licença

Este projeto está sob a licença MIT.

---

## 🙏 Agradecimentos

- [Hugging Face](https://huggingface.co/) - Biblioteca Transformers
- [Google Research](https://research.google/) - BERT
- [OpenAI](https://openai.com/) - GPT
- Comunidade de NLP

---

**💡 Dica**: Transformers são o presente e futuro do NLP. Invista tempo para dominar!

*Desenvolvido como parte do curso "Especialista em IA" - Módulo EAI_05*
