import onnxruntime as ort
import cv2
import numpy as np

model_path = "./RTMDet-m_640x640.onnx"
session = ort.InferenceSession(model_path)

origin = cv2.imread("./demo.jpg")
image = np.zeros((640, 640, 3))
image[107:532, :, :] = origin

input = np.float32(image)
input = input.transpose(2, 0, 1)
input = np.expand_dims(input, 0) / 255.0
input[:, 0, :, :] = (input[:, 0, :, :] - 0.406) / 0.225
input[:, 1, :, :] = (input[:, 1, :, :] - 0.456) / 0.224
input[:, 2, :, :] = (input[:, 2, :, :] - 0.485) / 0.229

outputs = session.run(None, {"input": input})
print(outputs[0].shape)
print(outputs[1].shape)
dets = outputs[0]
_, cnt, _ = dets.shape
print(cnt)

for i in range(cnt):
    print(dets[0, i, 0])
    cv2.rectangle(image, (int(dets[0, i, 0]), int(dets[0, i, 1])), (int(dets[0, i, 2]), int(dets[0, i, 3])), (0, 255, 0), 2)
cv2.imwrite("output.jpg", image)