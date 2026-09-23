# 🔢 DigitCheck — Reconhecimento de Sequências de Dígitos Manuscritos

O **DigitCheck** é um projeto de Visão Computacional desenvolvido para reconhecer **sequências de dígitos manuscritos em fotografias reais**.

O projeto começou com a classificação de um único dígito e evoluiu para um pipeline completo capaz de:

- localizar vários dígitos em uma mesma fotografia;
- separar os símbolos encontrados;
- transformar cada recorte para o formato esperado pela rede neural;
- classificar cada dígito de `0` a `9`;
- estimar a confiança de cada previsão;
- rejeitar previsões consideradas pouco confiáveis;
- reconstruir a sequência numérica na ordem correta;
- exportar o modelo treinado para **ONNX**;
- executar a inferência também fora do Google Colab.

🔗 **Aplicação Web:**  
https://digit-check--samuelnascime32.replit.app

📓 **Notebook no Google Colab:**  
https://colab.research.google.com/drive/1Ski2DJO8octt3UTjwwwq-7GtTPwEXuTo?usp=sharing

---

## 📌 Objetivo

O objetivo principal do projeto é reduzir a diferença entre um classificador tradicional treinado em bases como o **MNIST** e uma situação real, na qual os números aparecem em fotografias com:

- diferentes tamanhos;
- iluminação variável;
- inclinação;
- perspectiva;
- ruído;
- desfoque;
- sombras;
- traços fragmentados;
- vários dígitos na mesma imagem.

Por isso, o DigitCheck não é apenas uma CNN que recebe uma imagem `28×28`.

Existe um pipeline completo antes e depois da classificação.

---

## 🧠 Visão geral do funcionamento

O fluxo principal do DigitCheck é:

```text
Fotografia
   ↓
Correção de orientação
   ↓
Binarização e normalização
   ↓
Detecção de componentes
   ↓
Filtragem de ruídos
   ↓
Agrupamento de fragmentos
   ↓
Ordenação dos dígitos
   ↓
Recortes 28×28
   ↓
CNN
   ↓
TTA + análise de confiança
   ↓
Aceitação ou rejeição
   ↓
Sequência final
```

Cada etapa foi separada para permitir identificar se um erro veio da **segmentação**, da **classificação** ou da decisão de **confiança/rejeição**.

---

## 🔍 1. Segmentação da fotografia

Uma fotografia real não pode ser enviada diretamente para uma CNN treinada com MNIST. Primeiro é necessário localizar os dígitos.

O DigitCheck utiliza **OpenCV** para transformar a fotografia em regiões que possam ser analisadas individualmente.

### Binarização

A imagem é convertida para tons de cinza e recebe suavização com filtro Gaussiano.

Em seguida, o fundo da fotografia é estimado e normalizado para reduzir diferenças de iluminação.

O sistema utiliza limiarização automática com **Otsu** e também pode incorporar informação de cor quando existe um traço colorido suficientemente relevante.

O resultado é uma imagem binária em que o objetivo é deixar os traços dos dígitos destacados do fundo.

### Detecção de componentes

Depois da binarização, são encontrados contornos e componentes conectados.

Nem todo componente encontrado é um número. Pequenos pontos, bordas da folha, riscos e outros elementos podem aparecer como candidatos.

Por isso, o pipeline utiliza critérios como:

- área;
- altura;
- largura;
- proporção entre largura e altura;
- densidade de pixels;
- proximidade das bordas;
- aparência de linha ou ruído.

Componentes considerados incompatíveis com um dígito são descartados.

### Agrupamento de fragmentos

Um único algarismo pode ser dividido em vários componentes durante a binarização.

Isso é especialmente importante em números manuscritos com pequenos traços separados.

O DigitCheck verifica a distância e a relação espacial entre componentes e pode uni-los quando existe evidência de que pertencem ao mesmo dígito.

### Ordenação

No modo de sequência, as caixas encontradas são organizadas espacialmente para recuperar a ordem natural de leitura, da esquerda para a direita.

---

## 🖼️ 2. Preparação dos recortes

Depois que a região de cada dígito é localizada, o recorte ainda precisa ficar semelhante ao padrão usado durante o treinamento.

Cada símbolo passa pelas seguintes etapas:

1. remoção do espaço vazio desnecessário;
2. redimensionamento proporcional;
3. preservação da proporção original;
4. inserção em uma imagem preta de `28×28`;
5. centralização usando os momentos da imagem.

Assim, a entrada final da rede possui formato:

```text
28 × 28 × 1
```

O objetivo dessa etapa é aproximar uma fotografia real do formato encontrado no MNIST sem deformar o algarismo.

---

## 🧪 3. Base de dados e aumento sintético

O treinamento parte do **MNIST**, com os dígitos de `0` a `9`.

Entretanto, imagens do MNIST são muito mais limpas do que fotografias reais. Para reduzir essa diferença de domínio, o notebook cria versões sintéticas dos dígitos.

As transformações incluem:

- rotação de até aproximadamente ±20°;
- deslocamento;
- alteração de escala;
- pequenas mudanças de perspectiva;
- erosão e dilatação;
- desfoque Gaussiano;
- alteração de brilho e contraste;
- ruído;
- compressão JPEG;
- pequenas falhas no traço.

Essas variações simulam parte dos problemas que podem ocorrer quando um número é fotografado com um celular.

---

## 🏗️ 4. Modelos avaliados

Em vez de escolher uma arquitetura única sem comparação, o notebook treina **três experimentos controlados**.

| Experimento | Dados | Características |
|---|---|---|
| `01_baseline` | MNIST original | CNN compacta |
| `02_compacto_sintetico` | MNIST + sintéticos | CNN compacta com dados mais variados |
| `03_espacial_sintetico` | MNIST + sintéticos | CNN mais profunda, Batch Normalization e maior capacidade espacial |

### Arquitetura compacta

A versão compacta utiliza:

```text
Input 28×28×1
↓
Rescaling
↓
Data Augmentation
↓
Conv2D 16
↓
MaxPooling
↓
Conv2D 32
↓
MaxPooling
↓
Conv2D 64
↓
GlobalAveragePooling
↓
Dense 32
↓
Dropout
↓
Dense 10 + Softmax
```

### Arquitetura espacial

A arquitetura espacial preserva mais informação da posição dos traços:

```text
Input 28×28×1
↓
Rescaling
↓
Data Augmentation
↓
Conv2D 32
↓
Batch Normalization
↓
Conv2D 32
↓
MaxPooling
↓
Dropout
↓
Conv2D 64
↓
Batch Normalization
↓
Conv2D 64
↓
MaxPooling
↓
Dropout
↓
Flatten
↓
Dense 96
↓
Dropout
↓
Dense 10 + Softmax
```

Na execução utilizada no desenvolvimento, o modelo vencedor foi o:

```text
03_espacial_sintetico
```

com melhor desempenho na validação mista entre os três experimentos.

---

## 🏋️ 5. Treinamento

O treinamento utiliza:

- otimizador **Adam**;
- `sparse_categorical_crossentropy`;
- até **40 épocas**;
- `EarlyStopping`;
- `ReduceLROnPlateau`;
- `ModelCheckpoint`;
- salvamento do melhor modelo;
- seed fixa para melhorar a reprodutibilidade.

O limite de 40 épocas é apenas um teto.

Se a validação deixa de melhorar, o treinamento pode ser encerrado automaticamente antes disso.

---

## 📊 6. Seleção do melhor modelo

Os três modelos são comparados utilizando conjuntos de validação independentes.

Quando não existe quantidade suficiente de fotografias reais para uma calibração confiável, o score de seleção utiliza:

```text
40% validação MNIST
+
60% validação sintética
```

Quando existe um conjunto real de calibração adequado, esse domínio também pode entrar no critério de seleção.

O conjunto real de **teste** é mantido separado e não deve ser utilizado para escolher o modelo.

---

## 🎯 7. TTA e confiança da previsão

O sistema não utiliza somente a maior probabilidade da Softmax.

Durante a inferência é aplicado **Test-Time Augmentation (TTA)**, realizando pequenas variações sobre o mesmo recorte e comparando as respostas do modelo.

A decisão pode considerar:

- probabilidade da classe vencedora;
- diferença entre a primeira e a segunda classe mais provável;
- concordância entre as previsões do TTA;
- qualidade da segmentação.

Na configuração conservadora utilizada no projeto, foram considerados critérios como:

```text
Probabilidade mínima: 0.80
Margem mínima:        0.25
Concordância TTA:     0.75
Qualidade mínima:     0.35
```

Quando um dígito não passa pelos critérios, ele pode ser marcado para revisão em vez de ser aceito automaticamente.

---

## 🔄 8. Orientação da fotografia

Fotografias podem ser enviadas em orientações diferentes.

O pipeline possui suporte para:

```text
0°
90°
-90°
180°
```

Existe tanto a possibilidade de selecionar a rotação manualmente quanto utilizar análise automática em partes do pipeline.

Isso evita que uma fotografia correta seja classificada incorretamente apenas por estar girada.

---

## 🔢 9. Reconstrução da sequência

Depois de segmentar e classificar todos os recortes, as previsões são colocadas na ordem espacial dos símbolos.

Exemplo:

```text
Recortes encontrados:
[5] [9] [7] [8] [2] [3] [4] [2]

Resultado:
59782342
```

Nos testes registrados no notebook também foram reconhecidas sequências como:

```text
57
49
85
704
49704
0123456789
999
777
555
```

Esses exemplos ajudam a verificar não apenas a classificação individual, mas também a capacidade de localizar e ordenar múltiplos dígitos na mesma fotografia.

---

## 📈 10. Avaliação em três níveis

O projeto diferencia três problemas que muitas vezes são tratados como se fossem um só.

### Segmentação

Verifica se o sistema conseguiu localizar corretamente um candidato na fotografia.

### Classificação

Mede se a CNN reconheceu corretamente um recorte já segmentado.

### Resultado fim a fim

Avalia se todo o caminho:

```text
foto → segmentação → classificação → resultado
```

produziu a resposta correta.

O notebook também calcula métricas como:

- taxa de segmentação;
- acurácia de classificação condicional;
- acurácia bruta fim a fim;
- cobertura operacional;
- acurácia entre previsões aceitas;
- matriz de confusão.

Essa separação é importante porque uma CNN pode apresentar ótima acurácia em imagens já recortadas e ainda assim falhar em fotografias reais por problemas de localização ou segmentação.

---

## 📦 11. Exportação para ONNX

Depois da seleção do modelo final, o DigitCheck cria um grafo exclusivo de inferência.

As camadas aleatórias de `RandomTranslation`, `RandomRotation` e `RandomZoom` são utilizadas apenas durante o treinamento e são removidas do fluxo de inferência.

Antes da exportação, o notebook compara:

```text
Modelo Keras original
        ↓
Modelo Keras de inferência
        ↓
Modelo ONNX
```

O ONNX só é considerado válido se reproduzir as mesmas classes previstas pelo modelo Keras.

O arquivo gerado é:

```text
modelo_digitcheck_sequencia_v2.onnx
```

A entrada utilizada pelo modelo é:

```text
(None, 28, 28, 1)
```

e a saída contém as probabilidades das 10 classes:

```text
0 1 2 3 4 5 6 7 8 9
```

Na validação registrada no notebook, a diferença numérica máxima entre a saída Keras e ONNX ficou na ordem de `10⁻⁷`, mantendo as mesmas classes previstas.

---

## 🌐 12. Aplicação Web

O modelo também pode ser utilizado através da interface web do DigitCheck:

🔗 **https://digit-check--samuelnascime32.replit.app**

A aplicação foi criada para permitir o uso do modelo sem precisar interagir diretamente com o código do notebook.

O fluxo é:

```text
Selecionar fotografia
        +
Carregar modelo ONNX
        +
Carregar labels.txt
        ↓
Processar imagem
        ↓
Segmentar os dígitos
        ↓
Executar o modelo
        ↓
Exibir recortes e sequência reconhecida
```

O arquivo `labels.txt` deve representar as classes na mesma ordem utilizada pelo modelo:

```text
0
1
2
3
4
5
6
7
8
9
```

---

## ▶️ Como executar o projeto no Google Colab

### Opção 1 — Reproduzir o treinamento completo

1. Abra o notebook:

   **https://colab.research.google.com/drive/1Ski2DJO8octt3UTjwwwq-7GtTPwEXuTo?usp=sharing**

2. Faça uma cópia para o seu Google Drive, caso o Colab solicite.

3. No menu superior, selecione:

```text
Ambiente de execução → Executar tudo
```

4. Autorize a montagem do Google Drive quando solicitado.

5. O notebook irá:

```text
carregar o MNIST
→ criar dados sintéticos
→ preparar os conjuntos
→ treinar os três experimentos
→ comparar os modelos
→ selecionar o melhor
→ avaliar o classificador
→ preparar as funções de inferência
```

6. Aguarde o treinamento terminar.

Uma GPU do Colab é recomendada para reduzir o tempo de treinamento, mas o código também pode ser executado em CPU com maior tempo de processamento.

---

## 📷 Como testar uma fotografia no Colab

Depois que o modelo estiver treinado e carregado:

1. vá até a seção:

```text
17 — Demonstração com fotografias
```

2. mantenha:

```python
MODO_LEITURA = 'sequencia'
ROTACAO_FOTO = 0
ROI_MANUAL = None
QUANTIDADE_ESPERADA = None
```

3. execute a célula;

4. selecione uma ou mais fotografias;

5. o notebook exibirá:

- fotografia analisada;
- binarização;
- caixas encontradas;
- recortes `28×28`;
- probabilidades;
- concordância TTA;
- qualidade da segmentação;
- sequência final sugerida.

Caso a imagem esteja girada, altere:

```python
ROTACAO_FOTO = 90
```

ou:

```python
ROTACAO_FOTO = -90
```

ou:

```python
ROTACAO_FOTO = 180
```

e execute novamente.

---

## 🧠 Como testar diretamente o modelo ONNX no Colab

Após executar a etapa de exportação, vá até a célula:

```text
TESTE FINAL — SEQUÊNCIA USANDO O ONNX
```

Ela:

1. carrega `modelo_digitcheck_sequencia_v2.onnx`;
2. solicita o upload de uma fotografia;
3. utiliza a mesma segmentação da demonstração;
4. envia cada recorte ao ONNX Runtime;
5. mostra a classe e a probabilidade;
6. reconstrói a sequência final.

---

## 🗂️ Fotografias reais

O notebook também possui suporte para um conjunto real organizado em:

```text
DigitCheck_V3/
└── dados_reais/
    ├── treino/
    │   ├── 0/
    │   ├── 1/
    │   └── ...
    ├── calibracao/
    │   ├── 0/
    │   ├── 1/
    │   └── ...
    └── teste/
        ├── 0/
        ├── 1/
        └── ...
```

As fotografias de `teste` não devem ser usadas no treinamento ou na seleção do modelo.

Um arquivo compactado com imagens utilizadas nos testes pode acompanhar a entrega para facilitar a reprodução dos experimentos.

---

## 🧰 Tecnologias utilizadas

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Pillow**
- **ONNX**
- **tf2onnx**
- **ONNX Runtime**
- **Google Colab**
- **Replit**

---

## 📂 Arquivos principais da entrega

```text
DigitCheck/
├── DigitCheck_sequencia_estavel.ipynb
├── modelo_digitcheck_sequencia_v2.onnx
├── labels.txt
├── imagens_teste.zip
└── README.md
```

> O nome do arquivo `.zip` pode variar conforme a versão final enviada ao repositório.

---

## ⚠️ Limitações atuais

Apesar de funcionar bem em diversos exemplos, o sistema ainda depende de segmentação clássica por visão computacional.

Casos difíceis podem ocorrer quando existem:

- dígitos encostados;
- desenhos próximos aos números;
- iluminação muito desigual;
- baixo contraste;
- traços extremamente fragmentados;
- símbolos fora do conjunto `0–9`;
- perspectiva muito intensa.

Essas limitações são importantes porque ajudam a separar o desempenho do classificador do desempenho do sistema completo.

---

## 🚀 Próximas evoluções

A arquitetura foi construída de forma incremental.

Entre as próximas possibilidades estão:

- substituir ou complementar a segmentação clássica por detecção aprendida;
- incluir operadores matemáticos;
- reconhecer expressões completas;
- interpretar relações espaciais;
- gerar uma representação matemática estruturada;
- expandir a interface web.

O DigitCheck, portanto, serve como base para evoluir de um reconhecedor de dígitos para um sistema mais completo de reconhecimento matemático.

---

## 👨‍💻 Autor

**Samuel Barbosa**

Projeto desenvolvido como atividade acadêmica de Inteligência Artificial / Visão Computacional.
