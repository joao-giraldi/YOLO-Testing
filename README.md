# YOLO-Testing
Este repositório tem como objetivo centralizar estudos, anotações, códigos de teste e experimentos práticos relacionados à arquitetura YOLO (You Only Look Once), amplamente utilizada em tarefas de detecção de objetos em tempo real.

Vale notar que esse repositório **não incluí um dataset**.

# You Only Look Once (YOLO)

Resumidamente, trata-se de uma arquitetura de CNN (rede neural convolucional) voltada para a tarefa de detecção de objetos em imagens ou vídeos.

Essa abordagem (diferente de outras que dividem a detecção em etapas) o YOLO faz tudo de uma vez só, ou seja, com uma única passada (*forward pass*) pela imagem. Isso torna ele **muito mais rápido** sendo ideal para aplicações em tempo real.

## Como o YOLO funciona?

O YOLO divide a imagem de entrada em uma **grade (grid) e t**udo isso é feito em **um único pipeline**. Para cada célula da grade, ele:

1. **Prediz as *bounding* boxes (caixas delimitadoras)** — posições dos objetos.
2. **Atribui um score de confiança** para cada box.
3. **Classifica os objetos** dentro dessas boxes.

## Etapas da arquitetura YOLO

1. **Entrada (Input Layer):**
    - Imagem de tamanho fixo, geralmente 448x448x3 (altura, largura, canais RGB).
2. **Camadas Convolucionais + Pooling:**
    - 24 camadas convolucionais.
    - Alternância entre convoluções e camadas de max pooling.
    - Usadas para extrair **características espaciais** da imagem, como bordas, texturas, formas, etc.
3. **Camadas Totalmente Conectadas (Fully Connected Layers):**
    - 2 camadas densas após as convoluções.
    - A saída final é um vetor de previsão que representa:
        - **Bounding boxes**: coordenadas (x, y, w, h).
        - **Confiança**: quão provável é que haja um objeto ali.
        - **Classificação**: a classe mais provável daquele objeto.
4. **Grade (Grid) e Previsões:**
    - A imagem é dividida em uma **grade SxS** (por padrão, 7x7).
    - Para cada célula da grade:
        - A rede prevê **B bounding boxes** (por padrão, 2).
        - Para cada caixa: 5 valores (x, y, w, h, confiança).
        - E **C classes de objetos** (por exemplo, 20 para Pascal VOC).
    - A saída total da rede tem forma **S x S x (B×5 + C)**.
  
      ## Como usar o YOLO (yolov8) em python

### Instalação

```bash
pip install ultralytics
```

```python
from ultralytics import YOLO

# Carrega um modelo pré-treinado YOLOv8 (pode ser 'n', 's', 'm', 'l', 'x')
model = YOLO("yolov8n.pt")  # n = nano, mais leve e rápido
```

Cada modelo pré-treinado representa variações do tamanho e da complexidade do modelo, afetando desempenho em termos de FPS, uso de Memória e precisão na detecção (*mAP - mean Average Precision*)

Os prefixos significam:

- `yolov8n.pt` (Nano): Muito rápido com precisão baixa
- `yolov8s.pt` (Small): Rápido com precisão média
- `yolov8m.pt` (Medium): Tempo normal com precisão boa
- `yolov8l.pt` (Large): Lento com precisão muito boa
- `yolov8x.pt` (XLarge): Muito lento com precisão excelente

### Fazendo interferência em imagens

```python
# Faz a detecção na imagem
results = model("exemplo.jpg", show=True)  # show=True abre a imagem com os resultados
# Salva os resultados
results = model("exemplo.jpg", save=True)
```

### Fazendo detecção em vídeos ou webcam

```python
model.predict(source=0, show=True)  # 'source=0' é a webcam padrão
model.predict(source="video.mp4", save=True, show=True) # usando um vídeo
```

### Treinando com seu próprio dataset

```kotlin
dataset/
├── images/   //São as imagens brutas do dataset
│   ├── train/    //Imagens para treinar o modelo
│   └── val/      //Imagens usadas para validar o desempenho durante o treino (validação)
├── labels/  //Contém os arquivos de anotação, cada arquivo descreve os objetos presentes
│   ├── train/
│   └── val/
├── data.yaml //Arquivo de configuração principal do treino, ele informa ao modelo onde
							//estão os dados, quantas classes há, os nomes, etc...
```

```yaml
train: ./images/train
val: ./images/val

nc: 2  # Número de classes
names: ['cachorro', 'gato']  # Nomes das classes

```

E então você treina o YOLO assim:

```python
model = YOLO("yolov8n.pt")
model.train(data="data.yaml", epochs=50, imgsz=640)
```

Se aprofundando mais no comando `model.train()` temos:
```python
model.train(
    data="data.yaml",         # Caminho para o arquivo data.yaml
    epochs=100,               # Número de épocas (quantas vezes o modelo vê todas as imagens do conjunto de treino
    imgsz=640,                # Tamanho da imagem de entrada
    batch=16,                 # Tamanho do batch (quantas imagens o modelo processa de uma vez)
    device=0,                 # Define o dispositivo de treino ('0' = GPU, 'cpu' = CPU, '0,1' = múltiplas GPUs, etc...)
    name="meu_treino",        # Subpasta onde vai ser salva
    project="runs/train",     # Pasta principal onde vai ser salva
    lr0=0.01,                 # Taxa de aprendizado inicial
    patience=20,              # Early stopping após 20 épocas sem melhora
    augment=True,             # Ativa data augmentation
    cache=True,               # Usa cache em RAM para acelerar o carregamento
    rect=False,               # False = redimensiona imagem para quadrado (padrão YOLO)
    resume=False              # Continua o treino da onde parou
    val=True,                 # Realiza validação ao final de cada época
    verbose=True              # Mostra logs detalhados durante o treino
)
```
