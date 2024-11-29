import os
import time
import argparse
import torch
from ultralytics import YOLO

# Function to export YOLOv8 model to TensorRT
def export_to_tensorrt(model_path, output_dir, quant_mode):
    model = YOLO(model_path)  # Load YOLOv8 model
    
    print(f"Exporting {model_path} to TensorRT with {quant_mode} quantization...")
    export_args = {
        'device': '0',
        'optimize': True,
        'simplify': True,
        'int8': quant_mode == 'int8',
    }
    
    start_time = time.time()
    model.export(format='engine', imgsz=640, half=True, dynamic=True, project=output_dir, **export_args)
    end_time = time.time()

    export_time = end_time - start_time
    print(f"Export completed in {export_time:.2f} seconds.")
    return export_time

# Function to test inference speed on a sample image
def test_inference_speed(model_path, image_path):
    model = YOLO(model_path)  # Load TensorRT model
    
    print(f"Running inference on {image_path}...")
    start_time = time.time()
    results = model(image_path)
    end_time = time.time()

    inference_time = end_time - start_time
    print(f"Inference completed in {inference_time:.2f} seconds.")

    return inference_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 TensorRT Export and Test")
    parser.add_argument('--model', type=str, required=True, help="Path to YOLOv8 model (e.g., yolov8n.pt)")
    parser.add_argument('--output', type=str, default='./outputs', help="Output directory for TensorRT models")
    parser.add_argument('--quant_mode', type=str, choices=['fp16', 'int8'], default='fp16', help="Quantization mode for TensorRT export")
    parser.add_argument('--image', type=str, required=True, help="Path to sample image for inference test")
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Export model to TensorRT
    export_time = export_to_tensorrt(args.model, args.output, args.quant_mode)

    # Test inference speed
    tensorrt_model_path = os.path.join(args.output, 'engine', os.path.basename(args.model).replace('.pt', '.engine'))
    inference_time = test_inference_speed(tensorrt_model_path, args.image)

    print("\nSummary:")
    print(f"Model exported to TensorRT in {export_time:.2f} seconds.")
    print(f"Inference time: {inference_time:.2f} seconds.")
