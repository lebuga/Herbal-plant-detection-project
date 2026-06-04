from ultralytics import YOLO
import gradio as gr

# Load model
model = YOLO("YOLOv8_best.pt")

def detect(image):

    results = model.predict(
        source=image,
        conf=0.5
    )

    annotated_image = results[0].plot()

    detections = []

    for box in results[0].boxes:
        class_id = int(box.cls)
        confidence = float(box.conf)

        detections.append(
            f"{model.names[class_id]} ({confidence:.2f})"
        )

    if len(detections) == 0:
        detections.append("No plant detected")

    return annotated_image, "\n".join(detections)

demo = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="pil", height=150),
    outputs=[
        gr.Image(label="Detection Result",  height=150),
        gr.Textbox(label="Detected Classes",lines=1)
    ],
    title="🌿 Herbal Plant Detector",
    description="Upload an image to detect herbal plants in natural environments."
)

demo.launch()
