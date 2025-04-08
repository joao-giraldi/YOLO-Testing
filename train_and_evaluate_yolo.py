import os
from ultralytics import YOLO
import glob


def main():
    # Caminho dinâmico para os arquivos
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_YAML = os.path.join(SCRIPT_DIR, "dataset", "data.yaml")
    MODEL_PATH = os.path.join(SCRIPT_DIR, "yolov8m.pt")

    # Treinamento
    print("Iniciando treinamento do modelo YOLOv8...")
    model = YOLO(MODEL_PATH)

    model.train(
        data=DATA_YAML,
        epochs=50,
        save_period = 1,
        imgsz=640,
        batch=16,
        device="0",  # 0 = gpu, "cpu" = cpu
        name="detector_pedestres",
        project="runs/train",
        lr0=0.005,
        patience=20,
        augment=True,
        cache=False,
        rect=False,
        resume=False,
        val=True,
        verbose=True
    )

    # Avaliação
    print("\nAvaliando o modelo treinado no conjunto de teste...")
    train_dirs = glob.glob("runs/train/detector_pedestres*")
    latest_dir = max(train_dirs, key=os.path.getmtime)  # pega a mais recente

    best_model_path = os.path.join(latest_dir, "weights", "best.pt")
    model = YOLO(best_model_path)
    metrics = model.val(data=DATA_YAML, split="test")

    print("\nMétricas de avaliação:")
    for key, value in metrics.results_dict.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    # Predições visuais
    print("\nGerando predições visuais salvas em runs/predict/...")
    model.predict(
        source=os.path.join(SCRIPT_DIR, "dataset", "test", "images"),
        save=True,
        conf=0.3,
        iou=0.5,
        show=False
    )

    print("\nProcesso concluído com sucesso!")

if __name__ == "__main__":
    main()
