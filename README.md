# 🤖 Módulo 05 – NLP com Transformers

Este módulo apresenta os conceitos e aplicações práticas de modelos baseados em Transformers para Processamento de Linguagem Natural (PLN), com foco na utilização da biblioteca Hugging Face (`transformers`) e no uso de modelos pré-treinados para tarefas como análise de sentimentos e criação de chatbots.

---

## 📌 Objetivos

- Entender o funcionamento geral dos Transformers e seus benefícios para NLP
- Utilizar modelos pré-treinados via `pipeline` da Hugging Face
- Aplicar Transformers em tarefas de análise de sentimentos
- Criar um chatbot simples com LLM pré-treinado
- Executar inferências com modelos hospedados no Hugging Face Hub

---

## 🗂 Estrutura do Módulo

NLP_Transformers/
│
├── transformers_basico.ipynb
├── tokenizacao_transformers.ipynb
├── modelo_transforms.ipynb
├── classificacao_sentimentos_transformers.ipynb
└── chatbot_transformers.ipynb

---

## ⚙️ Tecnologias e Bibliotecas

- Python 3.10+
- Hugging Face Transformers
- Torch / TensorFlow (backend)
- Datasets personalizados ou prontos do Hugging Face
- Matplotlib e Pandas (para análise e visualização)

---

## 🧪 Projetos Desenvolvidos

- **Análise de Sentimentos com Transformers**  
  Uso do modelo `distilbert-base-uncased-finetuned-sst-2-english` através do `pipeline("sentiment-analysis")` para classificar sentimentos de frases em inglês.

- **Chatbot com LLM Pré-Treinado**  
  Implementação de um chatbot básico usando o modelo `mistralai/Mistral-7B-Instruct-v0.2` com prompt de entrada via pipeline (`text-generation`).

---

## 📈 Resultados e Conclusões

- O uso de Transformers pré-treinados elimina a necessidade de treinamento local em muitas tarefas.
- A API da Hugging Face simplifica drasticamente a inferência, mesmo com modelos de larga escala.
- Modelos como `DistilBERT` e `Mistral` oferecem excelente desempenho com pouca complexidade de uso.

> **Dica:** Para aplicações em produção, utilize nomes de modelos e versões específicas, garantindo reprodutibilidade e estabilidade.

---

## ✅ Conclusão

O módulo mostrou como aplicar com eficiência modelos baseados em Transformers em tarefas reais de NLP. Com as ferramentas da Hugging Face, é possível utilizar os modelos mais avançados do mundo com poucas linhas de código, abrindo portas para aplicações mais robustas em linguagem natural.

---

