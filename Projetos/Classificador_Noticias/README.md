# 📰 Classificador de Notícias com BERT

Sistema de classificação automática de notícias usando **DistilBERT fine-tunado** no dataset AG News. Classifica artigos em 4 categorias: World, Sports, Business, Sci/Tech.

---

## 🎯 Objetivo

Demonstrar **fine-tuning de Transformers** e comparar com NLP clássico:
- **EAI_04 (NLP Clássico)**: TF-IDF + SVM → ~87% accuracy
- **EAI_05 (Transformers)**: DistilBERT fine-tunado → ~94% accuracy
- **Ganho**: +7% accuracy com Transfer Learning

**Resultado**: Modelo deployado em Flask para classificação em tempo real.

---

## 🧠 Como Funciona

### Pipeline Completo

```
Notícia do Usuário
    ↓
Tokenização (DistilBERT Tokenizer)
    ↓
Modelo Fine-tunado (DistilBERT + Classifier Head)
    ↓
Softmax sobre 4 classes
    ↓
Categoria Predita + Confiança
```

### Exemplo de Uso

**Input**:
```
"Apple announces new iPhone with advanced AI features and 
improved camera system. The device will launch next month 
at a starting price of $999."
```

**Output**:
```
Categoria: Sci/Tech
Confiança: 96.8%
```

---

## 🏗️ Arquitetura do Modelo

### DistilBERT Base

```python
DistilBertModel(
  (embeddings): Embeddings(
    (word_embeddings): Embedding(30522, 768)
    (position_embeddings): Embedding(512, 768)
    (LayerNorm): LayerNorm((768,))
    (dropout): Dropout(p=0.1)
  )
  (transformer): Transformer(
    (layer): 6 x TransformerBlock(
      (attention): MultiHeadSelfAttention
      (sa_layer_norm): LayerNorm
      (ffn): FeedForward
      (output_layer_norm): LayerNorm
    )
  )
)
```

### Classifier Head (Adicionada)

```python
DistilBertForSequenceClassification(
  (distilbert): DistilBertModel(...)
  (pre_classifier): Linear(768 → 768)
  (classifier): Linear(768 → 4)  # 4 classes
  (dropout): Dropout(p=0.2)
)
```

**Total de Parâmetros**:
- DistilBERT base: 66M (frozen ou fine-tuned)
- Classifier head: ~600k (sempre treinado)

---

## 📊 Dataset - AG News

### Informações

**Fonte**: [AG News Corpus](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)

**Características**:
- 120.000 notícias de treino
- 7.600 notícias de teste
- 4 classes balanceadas
- Textos em inglês

### Classes

| Classe | Descrição | Exemplos |
|--------|-----------|----------|
| **World** (0) | Notícias internacionais | Política, guerras, diplomacia |
| **Sports** (1) | Esportes | Futebol, olimpíadas, campeonatos |
| **Business** (2) | Negócios | Economia, empresas, mercados |
| **Sci/Tech** (3) | Ciência/Tecnologia | IA, gadgets, pesquisas |

### Distribuição

```
Treino:
- World:    30.000 (25%)
- Sports:   30.000 (25%)
- Business: 30.000 (25%)
- Sci/Tech: 30.000 (25%)
Total:     120.000

Teste:
- World:    1.900 (25%)
- Sports:   1.900 (25%)
- Business: 1.900 (25%)
- Sci/Tech: 1.900 (25%)
Total:      7.600
```

### Exemplo de Dados

```python
{
  'text': 'Apple unveils new MacBook Pro with M3 chip',
  'label': 3  # Sci/Tech
}

{
  'text': 'Federal Reserve raises interest rates by 0.25%',
  'label': 2  # Business
}

{
  'text': 'Lakers defeat Celtics in overtime thriller',
  'label': 1  # Sports
}
```

---

## 🚀 Como Usar

### 1. Instalação

```bash
# Clonar repositório
git clone https://github.com/usuario/classificador-noticias-bert.git
cd classificador-noticias-bert

# Criar ambiente
conda create -n news_clf python=3.9
conda activate news_clf

# Instalar dependências
pip install -r requirements.txt
```

### 2. Treinar Modelo (Opcional)

O modelo já vem pré-treinado, mas você pode retreinar:

```bash
# Executar notebook
jupyter notebook notebook/finetuning_bert_noticias.ipynb

# Ou via Python
python scripts/train.py
```

**Tempo de treino**: ~1-2 horas (GPU) ou ~8-12 horas (CPU)

### 3. Executar Aplicação Flask

```bash
python app.py
```

**Acesse**: http://localhost:5000

### 4. Usar Interface

1. Cole ou digite uma notícia
2. Clique em **"Classificar"**
3. Veja categoria + confiança

---

## 📁 Estrutura do Projeto

```
Classificador_Noticias/
├── README.md (este arquivo)
├── AGENT_CONTEXT.md
├── app.py                      # 🌐 Backend Flask
├── requirements.txt            # 📦 Dependências
│
├── notebook/
│   └── finetuning_bert_noticias.ipynb  # 📓 Treinamento
│
├── models/
│   └── bert_news_classifier/   # 💾 Modelo fine-tunado
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer_config.json
│       └── vocab.txt
│
├── data/
│   ├── train.csv               # Dataset de treino
│   └── test.csv                # Dataset de teste
│
├── templates/                  # 🖼️ Interface web
│   ├── index.html
│   └── resultado.html
│
└── static/
    └── css/
        └── style.css           # 🎨 Estilos
```

---

## 🌐 Aplicação Flask

### Backend (app.py)

```python
from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# Carregar modelo fine-tunado
MODEL_PATH = 'models/bert_news_classifier'
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
modelo = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
modelo.eval()

# Mapeamento de classes
CLASSES = {
    0: 'World',
    1: 'Sports',
    2: 'Business',
    3: 'Sci/Tech'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    texto = request.form.get('news_text', '').strip()
    
    if not texto:
        return render_template('index.html', 
                             error="Por favor, insira uma notícia.")
    
    # Tokenizar
    inputs = tokenizer(
        texto,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        padding=True
    )
    
    # Predição
    with torch.no_grad():
        outputs = modelo(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_class].item()
    
    # Resultado
    categoria = CLASSES[pred_class]
    confianca = f"{confidence * 100:.2f}%"
    
    return render_template('resultado.html',
                         news_text=texto,
                         categoria=categoria,
                         confianca=confianca,
                         todas_probs={CLASSES[i]: f"{probs[0][i].item()*100:.2f}%" 
                                     for i in range(4)})

if __name__ == '__main__':
    app.run(debug=True)
```

### Frontend (templates/index.html)

```html
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Classificador de Notícias</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="container">
        <h1>📰 Classificador de Notícias com BERT</h1>
        <p class="subtitle">Classifica notícias em: World, Sports, Business, Sci/Tech</p>
        
        <form method="POST" action="/classify">
            <label for="news_text">Cole ou digite a notícia:</label>
            <textarea 
                id="news_text" 
                name="news_text" 
                rows="10" 
                placeholder="Ex: Apple unveils new iPhone with advanced AI features..."
                required
            ></textarea>
            
            <button type="submit">Classificar</button>
        </form>
        
        {% if error %}
        <p class="error">{{ error }}</p>
        {% endif %}
        
        <div class="examples">
            <h3>Exemplos para testar:</h3>
            <button onclick="fillExample(0)">World</button>
            <button onclick="fillExample(1)">Sports</button>
            <button onclick="fillExample(2)">Business</button>
            <button onclick="fillExample(3)">Sci/Tech</button>
        </div>
    </div>
    
    <script>
        const examples = [
            "UN Security Council holds emergency meeting to discuss escalating tensions in the Middle East.",
            "Lakers defeat Celtics 108-102 in overtime thriller at TD Garden.",
            "Federal Reserve raises interest rates by 0.25% to combat inflation.",
            "Google announces breakthrough in quantum computing with new processor."
        ];
        
        function fillExample(idx) {
            document.getElementById('news_text').value = examples[idx];
        }
    </script>
</body>
</html>
```

---

## 📊 Performance e Resultados

### Métricas de Avaliação

**Dataset de Teste (7.600 amostras)**:

```
Accuracy: 94.2%

              precision    recall  f1-score   support

       World       0.93      0.95      0.94      1900
      Sports       0.97      0.98      0.98      1900
    Business       0.93      0.91      0.92      1900
    Sci/Tech       0.94      0.93      0.93      1900

    accuracy                           0.94      7600
   macro avg       0.94      0.94      0.94      7600
weighted avg       0.94      0.94      0.94      7600
```

### Matriz de Confusão

```
                 Predito
              W    S    B    T
Real  World [1805  12   58   25]
      Sports[  8 1862   15   15]
      Busi  [ 72   11 1729   88]
      Tech  [ 31   18   93 1758]
```

**Análise**:
- ✅ **Sports**: Melhor performance (98% recall)
- ✅ **World**: Alta precision (93%)
- ⚠️ **Business ↔ Sci/Tech**: Maior confusão (empresas de tecnologia)

---

## 📈 Comparação: Clássico vs Transformers

### Mesma Tarefa, Modelos Diferentes

| Métrica | TF-IDF + SVM (EAI_04) | DistilBERT (EAI_05) | Ganho |
|---------|----------------------|---------------------|-------|
| **Accuracy** | 87.3% | 94.2% | **+6.9%** |
| **Precision** | 0.87 | 0.94 | +0.07 |
| **Recall** | 0.87 | 0.94 | +0.07 |
| **F1-Score** | 0.87 | 0.94 | +0.07 |
| **Treino** | 5 min (CPU) | 1.5h (GPU) | -95x |
| **Inferência** | <10ms | ~50ms | -5x |
| **Tamanho** | 5 MB | 260 MB | -52x |

### Quando Usar Cada Um?

**TF-IDF + SVM** (Clássico):
- ✅ Baseline rápido
- ✅ Recursos limitados
- ✅ Poucos dados (<10k)
- ✅ Deploy simples

**DistilBERT** (Transformers):
- ✅ Accuracy crítica
- ✅ GPU disponível
- ✅ Muitos dados (>50k)
- ✅ Estado-da-arte necessário

---

## 🔍 Casos de Teste

### Caso 1: Notícia Clara (Sci/Tech)

**Input**:
```
"OpenAI releases GPT-4, the most advanced language model to date, 
capable of processing images and text with unprecedented accuracy."
```

**Output**:
```
Categoria: Sci/Tech
Confiança: 98.7%

Distribuição:
- Sci/Tech: 98.7%
- Business: 1.1%
- World:    0.2%
- Sports:   0.0%
```

---

### Caso 2: Notícia Ambígua (Business/Tech)

**Input**:
```
"Tesla stock surges 15% after Elon Musk announces new AI-powered 
self-driving features coming to all vehicles by end of year."
```

**Output**:
```
Categoria: Business
Confiança: 67.3%

Distribuição:
- Business: 67.3%
- Sci/Tech: 31.2%
- World:    1.3%
- Sports:   0.2%
```

**Análise**: Modelo hesita entre Business (ações) e Tech (IA), mas escolhe corretamente Business pelo contexto de mercado.

---

### Caso 3: Notícia Esportiva

**Input**:
```
"Cristiano Ronaldo scores hat-trick as Al-Nassr defeats Al-Hilal 
4-2 in Saudi Pro League derby."
```

**Output**:
```
Categoria: Sports
Confiança: 99.5%

Distribuição:
- Sports:   99.5%
- World:    0.3%
- Business: 0.1%
- Sci/Tech: 0.1%
```

---

## 💻 Detalhes Técnicos

### Hiperparâmetros de Fine-tuning

```python
# Treinamento
learning_rate = 2e-5
batch_size = 16
epochs = 3
weight_decay = 0.01
warmup_steps = 500

# Optimizer
AdamW(lr=2e-5, eps=1e-8)

# Scheduler
get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=total_steps
)
```

### Data Augmentation (Opcional)

```python
# Back-translation
# Original: "Apple releases new iPhone"
# EN → PT: "Apple lança novo iPhone"
# PT → EN: "Apple launches new iPhone"

# Paraphrasing com T5
# Original: "Stock market crashes"
# T5: "Equity markets experience sharp decline"
```

---

## 🔮 Melhorias Futuras

### Modelo
- [ ] Testar BERT-base (em vez de DistilBERT)
- [ ] Ensemble (DistilBERT + RoBERTa)
- [ ] Multi-task learning (classificação + NER)
- [ ] Usar modelos maiores (BERT-large, DeBERTa)

### Dados
- [ ] Adicionar mais classes (Entertainment, Health)
- [ ] Dataset multilíngue (português, espanhol)
- [ ] Active Learning (retreinar com feedbacks)
- [ ] Balanceamento por sub-tópicos

### Aplicação
- [ ] API REST (FastAPI)
- [ ] Upload de arquivo CSV (batch)
- [ ] Explicabilidade (LIME, SHAP)
- [ ] Cache de predições (Redis)
- [ ] Deploy em cloud (Heroku, AWS)

### UX
- [ ] Confiança visual (barra de progresso)
- [ ] Histórico de classificações
- [ ] Exportar resultados (JSON, CSV)
- [ ] Dark mode

---

## 🤝 Contribuindo

Contribuições são bem-vindas!

1. Fork o repositório
2. Crie branch (`git checkout -b feature/melhoria`)
3. Commit mudanças
4. Push para branch
5. Abra Pull Request

---

## 📖 Recursos Adicionais

### Papers
- [DistilBERT](https://arxiv.org/abs/1910.01108) (Sanh et al., 2019)
- [BERT](https://arxiv.org/abs/1810.04805) (Devlin et al., 2018)

### Datasets Similares
- [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/)
- [BBC News](https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive)
- [Reuters-21578](https://archive.ics.uci.edu/ml/datasets/reuters-21578+text+categorization+collection)

### Ferramentas
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Datasets Library](https://huggingface.co/docs/datasets)
- [BERT Viz](https://github.com/jessevig/bertviz) - Visualizar atenção

---

## 📝 Citação

```
@misc{classificador_noticias_bert_2026,
  author = {Carlos Henrique Bamberg Marques},
  title = {Classificador de Notícias com DistilBERT Fine-tunado},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/usuario/classificador-noticias-bert}
}
```

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
- [AG News](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html) - Dataset
- Comunidade de NLP

---

**💡 Dica**: Fine-tuning é onde Transformers brilham! Este projeto mostra o poder real.

*Desenvolvido como parte do curso "Especialista em IA" - Módulo EAI_05*
