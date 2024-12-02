import torch
from ultralytics import YOLO
import time
import tensorrt as trt
import onnx

# Step 1: YOLOv8 모델 로드 및 TensorRT 엔진 변환
def export_to_tensorrt(model_path, export_path, quant_mode="fp16"):
    """
    YOLOv8 모델을 TensorRT 엔진으로 변환하는 함수.

    :param model_path: YOLOv8 PyTorch 모델 경로 (.pt 파일)
    :param export_path: TensorRT 엔진 파일 저장 경로
    :param quant_mode: 양자화 모드 ("fp16" 또는 "int8")
    """
    print(f"Loading YOLOv8 model from {model_path}...")
    model = YOLO(model_path)
    print("Exporting model to TensorRT engine...")
    model.export(format="engine", device="0", dynamic=True, half=(quant_mode == "fp16"))
    print(f"TensorRT engine saved at {export_path}")

# Step 2: TensorRT 엔진 추론
def run_inference(trt_model_path, input_image_path):
    """
    TensorRT 엔진으로 추론 시간을 측정하는 함수.

    :param trt_model_path: TensorRT 엔진 경로 (.engine 파일)
    :param input_image_path: 입력 이미지 경로
    """
    print(f"Loading TensorRT engine from {trt_model_path}...")
    model = torch.load(trt_model_path)
    print(f"Running inference on {input_image_path}...")

    # 추론 시간 측정
    start_time = time.time()
    results = model(input_image_path)
    end_time = time.time()

    print(f"Inference completed. Time taken: {end_time - start_time:.2f} seconds")
    return results

# Step 3: 실행
if __name__ == "__main__":
    # 모델 및 파일 경로 설정
    yolov8_model_path = "yolov8n.pt"
    trt_export_path = "yolov8n_fp16.engine"
    test_image_path = "sample.jpg"

    # TensorRT 엔진 생성
    export_to_tensorrt(yolov8_model_path, trt_export_path, quant_mode="fp16")

    # 추론 실행
    results = run_inference(trt_export_path, test_image_path)

    # 결과 출력
    print("Inference Results:")
    print(results)
