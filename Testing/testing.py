from ultralytics import YOLO

# --- Paths to models ---
models = {
    "YOLOv8n (COCO)": "../Weights/yolov8n.pt",
    "Custom Model 1": "../Weights/GoalLine_V1.pt",
    "Custom Model 2": "../Weights/GoalLine_V2.pt"
}

# --- Dataset config file ---
data_yaml = "data.yaml"   # e.g., "data.yaml"

# --- Evaluation parameters ---
test_split = "test"   # 'val' or 'test'
imgsz = 640
conf = 0.75
iou = 0.5
device = 'cpu'            # change to 'cpu' if no GPU
class_filter = None   # or [0] if you only want class "ball"

# --- Run evaluation for each model ---
results_summary = []

for name, path in models.items():
    print(f"\n🔍 Evaluating {name}...")
    model = YOLO(path)

    results = model.val(
        data=data_yaml,
        split=test_split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        classes=class_filter,
        verbose=False
    )

    results_summary.append({
        "Model": name,
        "mAP@0.5": results.box.map50,
        "mAP@0.5:0.95": results.box.map
    })

# --- Print comparison table ---
print(f"\n📊 Comparison Results at conf {conf}:")
print(f"{'Model':<20} {'mAP@0.5':<12} {'mAP@0.5:0.95':<12}")
print("-" * 45)
for r in results_summary:
    print(f"{r['Model']:<20} {r['mAP@0.5']:<12.4f} {r['mAP@0.5:0.95']:<12.4f}")
