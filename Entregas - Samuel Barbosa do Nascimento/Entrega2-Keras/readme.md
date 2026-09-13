# 🔢 DigitCheck

### Reconhecimento de dígitos manuscritos com TensorFlow e Keras

O **DigitCheck** é um projeto de inteligência artificial desenvolvido para reconhecer números manuscritos. O modelo utiliza uma **Rede Neural Convolucional (CNN)** treinada com o dataset público **MNIST**, disponível diretamente no Keras.

O projeto inclui treinamento, escolha automática do melhor modelo, gráficos de aprendizagem, matriz de confusão, análise dos erros e uma demonstração com fotografias externas.

---

## 🚀 Acesse o projeto

### [▶️ Abrir o DigitCheck no Google Colab](https://colab.research.google.com/drive/1cdqSufTnfVgh9sIaqza36T6wTG33eHMi?usp=sharing)

O notebook pode ser executado diretamente no navegador. Para editar ou treinar o modelo, faça uma cópia para o seu Google Drive.

---

## 🎯 Objetivo

Criar um protótipo capaz de:

- reconhecer dígitos manuscritos de **0 a 9**;
- indicar a confiança de cada previsão;
- encaminhar resultados duvidosos para revisão;
- salvar automaticamente o melhor modelo treinado;
- testar o sistema com imagens externas.

---

## 🧰 Tecnologias utilizadas

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy e Pandas
- Matplotlib e Seaborn
- Scikit-learn
- Google Colab

---

## 🧠 Modelo

Foi utilizada uma **CNN compacta**, adequada às imagens de baixa resolução do MNIST.

| Característica | Resultado |
|---|---:|
| Resolução de entrada | 28 × 28 |
| Classes | 10 |
| Parâmetros | 25.706 |
| Tamanho do modelo | 0,34 MB |
| Melhor época | 28 |
| Tempo de treinamento | 120,66 segundos |
| Inferência mediana | 11,38 ms |

---

## 📊 Principais resultados

O modelo foi avaliado em **10.000 imagens do conjunto de teste do MNIST**.

| Métrica | Resultado |
|---|---:|
| Acurácia geral | **98,19%** |
| Acurácia das previsões aceitas | **99,33%** |
| Cobertura das previsões aceitas | **97,27%** |
| Casos encaminhados para revisão | 273 |

A matriz de confusão apresentou forte concentração na diagonal principal, indicando que a maioria dos dígitos foi classificada corretamente. Os principais erros ocorreram entre números com formatos manuscritos semelhantes, como **9 e 3**, **7 e 2**, **6 e 8** e **8 e 9**.

---

## 📸 Teste com fotografias reais

O modelo também foi testado com fotografias de números escritos em uma folha pautada. Nesse cenário, nenhuma sequência foi reconhecida completamente, resultando em **0% de acurácia exata por sequência**.

O principal problema ocorreu na etapa de segmentação. Durante a binarização, as linhas do caderno também foram destacadas e interpretadas como possíveis algarismos. Com isso, o modelo recebeu recortes contendo linhas, partes incompletas dos números ou outros elementos que não existiam nas imagens de treinamento.

Alguns dígitos isolados foram reconhecidos corretamente, mostrando que o classificador aprendeu o padrão do MNIST. Entretanto, o produto completo não conseguiu generalizar satisfatoriamente para fotografias reais.

---

## ⚠️ Limitações identificadas

- O MNIST contém dígitos isolados e centralizados.
- As fotografias possuem linhas, sombras, rotação e perspectiva.
- O algoritmo pode confundir linhas do papel com números.
- O modelo não possui uma classe chamada “não é um dígito”.
- Uma confiança elevada não garante que a previsão esteja correta.

Esse resultado demonstra que uma boa acurácia no conjunto de teste não garante, sozinha, o funcionamento da solução em um cenário real.

---

## 💡 Possíveis melhorias

- remover linhas do papel antes da segmentação;
- corrigir automaticamente a rotação e a perspectiva;
- treinar com fotografias reais;
- aplicar aumento de dados com ruídos e variações de iluminação;
- implementar um mecanismo de rejeição para recortes inválidos;
- utilizar modelos próprios para reconhecimento de sequências.

---

## ✅ Conclusão

O DigitCheck cumpriu o objetivo acadêmico de construir, treinar, salvar e avaliar um modelo de inteligência artificial com um dataset público do Keras.

O modelo apresentou **98,19% de acurácia no MNIST** com baixo consumo de recursos. A demonstração externa revelou uma limitação importante: o classificador funciona bem em imagens semelhantes às utilizadas no treinamento, mas ainda precisa de melhorias para reconhecer sequências em fotografias reais.

---

**Autor:** Samuel Barbosa do Nascimento  
**Engenharia da Computação — UFG**
