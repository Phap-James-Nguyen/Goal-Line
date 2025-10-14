from ultralytics import YOLO  # Import YOLOv8 API

# Load a pre-trained YOLOv8 small model
model = YOLO("yolov8s.pt")

# Train the model on your soccer ball dataset
model.train(
    data="/home/dperezs/Workspace/SeniorDesign/CustomV2/data.yaml",         # Path to the dataset YAML
    epochs=50,                # Train for up to 50 epochs
    imgsz=640,                # Input image resolution
    batch=16,                 # Number of images per training batch
    patience=7,               # Stop early if val metric doesn't improve for 7 epochs
    lr0=0.001,                # Initial learning rate
    weight_decay=0.0005,      # Regularization to help generalization
    optimizer="SGD",          # Optimizer choice: SGD for fine-tuning
    single_cls=True,          # Treat all objects as one class
    project="runs/train",     # Base directory for logs and weights
    name="ballV2.0",            # Subfolder name for this experiment
    val=True                  # Run validation during training
)

# Evaluate the model after training ends
metrics = model.val()  # This uses the best weights by default
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
