import onnxruntime as ort
import cv2
import numpy as np

model_path = "./RTMPose-l_384_288.onnx"
session = ort.InferenceSession(model_path)

image = cv2.imread("./demo.jpg")
image = image[:, 160+60: 640 - 161+60, :]


image = cv2.resize(image, (288, 384), interpolation=cv2.INTER_LINEAR)
input = np.float32(image)
input = input.transpose(2, 0, 1)
input = np.expand_dims(input, 0) / 255.0
input[:, 0, :, :] = (input[:, 0, :, :] - 0.406) / 0.225
input[:, 1, :, :] = (input[:, 1, :, :] - 0.456) / 0.224
input[:, 2, :, :] = (input[:, 2, :, :] - 0.485) / 0.229


outputs = session.run(None, {"input": input})

print(outputs)

for i in range(133):
    x = np.argmax(outputs[0][0, i, :]) / 2.0
    y = np.argmax(outputs[1][0, i, :]) / 2.0
    cv2.circle(image, (int(x), int(y)), 1, 255, 4)
cv2.imwrite("output.jpg", image)
