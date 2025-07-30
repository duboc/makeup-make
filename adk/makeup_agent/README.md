# Assistente de Maquiagem Boticário 💄

Um agente inteligente que ajuda clientes a encontrar o tom perfeito de base e outros produtos de maquiagem da Boticário através de análise de imagem usando o sistema de Artifacts do ADK.

## 🎯 Funcionalidades

- **Análise de Tom de Pele**: Detecta automaticamente o tom de pele através de foto
- **Detecção de Subtom**: Identifica se o subtom é quente, frio ou neutro
- **Recomendações Personalizadas**: Sugere produtos específicos da Boticário
- **Dicas de Aplicação**: Fornece dicas de como aplicar os produtos
- **Interface Amigável**: Conversa natural em português brasileiro com emojis
- **Persistência com Artifacts**: Usa o sistema de Artifacts do ADK para salvar imagens

## 📋 Pré-requisitos

1. **API de Análise**: A API deve estar rodando em `http://localhost:9090`
   ```bash
   # Na raiz do projeto
   python app.py
   ```

2. **Configuração do Runner com ArtifactService**:
   ```python
   from google.adk.runners import Runner
   from google.adk.artifacts import InMemoryArtifactService
   from google.adk.sessions import InMemorySessionService
   from adk.makeup_agent import root_agent

   # Configure o serviço de artifacts
   artifact_service = InMemoryArtifactService()
   session_service = InMemorySessionService()

   # Crie o runner com o artifact service
   runner = Runner(
       agent=root_agent,
       app_name="makeup_assistant",
       session_service=session_service,
       artifact_service=artifact_service  # IMPORTANTE!
   )
   ```

## 🚀 Como Usar

### 1. Com Imagem
O usuário pode enviar uma foto do rosto e o agente:
- Salva a imagem como artifact automaticamente
- Carrega o artifact para análise
- Detecta o tom de pele e subtom
- Recomenda produtos da Boticário
- Fornece dicas personalizadas

### 2. Sem Imagem
Se o usuário não enviar imagem, o agente:
- Pede educadamente uma foto
- Explica como tirar uma boa foto
- Oferece dicas sobre iluminação

## 🛠️ Arquitetura com Artifacts

```
Usuário envia imagem
    ↓
Callback salva como artifact
    ↓
Agente usa analyze_skin_tone_tool
    ↓
Tool carrega artifact
    ↓
Tool chama API com imagem
    ↓
API retorna análise
    ↓
Usuário recebe recomendações
```

## 📁 Estrutura

```
adk/makeup_agent/
├── __init__.py      # Exporta o agente
├── agent.py         # Agente principal, callback e tool
└── README.md        # Este arquivo
```

## 🔧 Componentes Principais

### 1. `_save_uploaded_image_as_artifact`
Callback assíncrono que:
- Extrai imagem do user_content
- Cria um artifact usando `types.Part.from_bytes()`
- Salva como "user_photo.jpg" usando `save_artifact()`

### 2. `analyze_skin_tone_tool`
Tool assíncrona que:
- Carrega o artifact da imagem usando `load_artifact()`
- Extrai os bytes da imagem
- Chama a API de análise
- Retorna recomendações formatadas

### 3. `makeup_assistant`
Agente principal configurado para:
- Usar artifacts ao invés de state
- Ser amigável e profissional
- Responder em português brasileiro
- Focar nos produtos Boticário

## 📝 Exemplo de Resposta

```
✨ **Análise do seu tom de pele concluída!** ✨

**Cor detectada:** #d4a574
**Subtom:** Quente (confiança: 85%)

🎯 **Produtos recomendados da Boticário:**

**🌟 Combinações Excelentes:**
• **Make B. Base Líquida Mate Salicylic 30g**
  Tom 140: Claro com subtom neutro
  Compatibilidade: 95%

💄 **Dicas de uso:**
• Aplique a base com movimentos circulares
• Use um primer antes para durabilidade
• Seu subtom quente combina com blushes pêssego
```

## 🐛 Tratamento de Erros

O agente trata diversos cenários:
- Artifact service não configurado
- Sem imagem enviada
- Pele não detectada
- API offline ou timeout
- Erros de processamento

## ⚠️ Importante

**O ArtifactService DEVE ser configurado no Runner!** Sem ele, o agente não conseguirá salvar ou carregar imagens e retornará erros.

## 🔍 Logs

Configure o nível de log para debug:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Isso mostrará:
- Salvamento/carregamento de artifacts
- Chamadas à API
- Processamento de imagens
- Erros detalhados
