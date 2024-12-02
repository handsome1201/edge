import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os

# TensorRT 엔진 생성 및 INT8 양자화 수행 함수
def build_int8_engine(onnx_model_path, engine_file_path, calibration_dataset):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB workspace

    # INT8 Calibration
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = trt.IInt8Calibrator()

    # ONNX 모델 로드
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_model_path, 'rb') as model:
        parser.parse(model.read())

    # TensorRT 엔진 생성
    with builder.build_engine(network, config) as engine, open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())
    print(f"INT8 TensorRT engine saved at {engine_file_path}")

# TensorRT 엔진 로드 및 추론 수행 함수
def run_inference(trt_model_path, input_image_path):
    """
    TensorRT 엔진으로 추론을 수행하는 함수.

    :param trt_model_path: TensorRT 엔진 경로 (.engine 파일)
    :param input_image_path: 입력 이미지 경로
    """
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    print(f"Loading TensorRT engine from {trt_model_path}...")

    # TensorRT 엔진 로드
    with open(trt_model_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # 입력 및 출력 정보 가져오기
    input_shape = tuple(engine.get_binding_shape(0))  # (batch_size, channels, height, width)
    output_shape = tuple(engine.get_binding_shape(1))  # Output shape
    input_dtype = trt.nptype(engine.get_binding_dtype(0))
    output_dtype = trt.nptype(engine.get_binding_dtype(1))

    print(f"Input shape: {input_shape}, Output shape: {output_shape}")

    # 입력 이미지 전처리
    image = cv2.imread(input_image_path)
    resized_image = cv2.resize(image, (input_shape[2], input_shape[3]))
    input_image = resized_image.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_image = np.expand_dims(input_image, axis=0).astype(input_dtype)

    # Device 메모리 할당
    d_input = cuda.mem_alloc(input_image.nbytes)
    d_output = cuda.mem_alloc(np.prod(output_shape) * np.dtype(output_dtype).itemsize)
    bindings = [int(d_input), int(d_output)]

    # Stream 생성
    stream = cuda.Stream()

    # 입력 데이터를 Device로 복사
    cuda.memcpy_htod_async(d_input, input_image, stream)

    # 추론 실행
    start_time = time.time()
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    end_time = time.time()

    # 결과를 Host로 복사
    output = np.empty(output_shape, dtype=output_dtype)
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()

    print(f"Inference completed. Time taken: {end_time - start_time:.2f} seconds")
    return output

# 실행 코드
if __name__ == "__main__":
    # 모델 및 파일 경로 설정
    onnx_model_path = "yolov8n.onnx"  # ONNX 모델 파일 경로
    trt_engine_path = "yolov8n_int8.engine"  # TensorRT INT8 엔진 파일 경로
    calibration_dataset = "coco128"  # Calibration 데이터셋 경로

    # INT8 양자화 엔진 생성
    if not os.path.exists(trt_engine_path):
        build_int8_engine(onnx_model_path, trt_engine_path, calibration_dataset)

    # 추론 실행
    test_image_path = "sample.jpg"  # 입력 이미지 경로
    results = run_inference(trt_engine_path, test_image_path)

    # 결과 출력
    print("Inference Results:")
    print(results)
