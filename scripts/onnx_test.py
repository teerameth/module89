import glob
import onnx
onnx_model = onnx.load("../models/chessboard.onnx")
print(onnx.helper.printable_graph(onnx_model.graph))