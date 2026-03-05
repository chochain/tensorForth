import tensorflow as tf
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2

def create_mnist_cnn_graphdef(log_dir="./logs/mnist_cnn_final"):
    # Define the entire architecture in pbtxt format
    mnist_pbtxt = """
    # --- INPUT ---
    node {
      name: "input"
      op: "Placeholder"
      attr { key: "dtype" value { type: DT_FLOAT } }
      attr { key: "shape" value { shape { dim { size: 50 } dim { size: 28 } dim { size: 28 } dim { size: 1 } } } }
    }
    # --- CONV LAYER 1 ---
    node {
      name: "conv1/weight"
      op: "VariableV2"
      attr { key: "dtype" value { type: DT_FLOAT } }
      attr { key: "value" value { tensor { dtype: DT_FLOAT tensor_shape { dim { size: 3 } dim { size: 3 } dim { size: 1 } dim { size: 10 } } float_val: 0.2 } } }
    }
    node {
      name: "conv1/bias"
      op: "BiasAdd"
      attr { key: "dtype" value { type: DT_FLOAT } }
      attr { key: "value" value { tensor { dtype: DT_FLOAT tensor_shape { dim { size: 10 } } float_val: 0.5 } } }
    }
    node {
      name: "conv1/Conv2D"
      op: "Conv2D"
      input: "input"
      input: "conv1/weight"
      input: "conv1/bias"
      attr { key: "T" value { type: DT_FLOAT } }
      attr { key: "strides" value { list { i: 1 i: 1 i: 1 i: 1 } } }
      attr { key: "padding" value { s: "SAME" } }
      attr { key: "_output_shape" value { shape { dim { size: 50 } dim { size: 28 } dim { size: 28 } dim { size: 1 } } } }
    }
    node {
      name: "conv1/MaxPool"
      op: "MaxPool"
      input: "conv1/Conv2D"
      attr { key: "ksize" value { list { i: 1 i: 2 i: 2 i: 1 } } }
      attr { key: "strides" value { list { i: 1 i: 2 i: 2 i: 1 } } }
      attr { key: "padding" value { s: "SAME" } }
      attr { key: "_output_shape" value { shape { dim { size: 50 } dim { size: 14 } dim { size: 14 } dim { size: 1 } } } }
    }
    node {
      name: "conv1/Relu"
      op: "Relu"
      input: "conv1/MaxPool"
      attr { key: "_output_shape" value { shape { dim { size: 50 } dim { size: 14 } dim { size: 14 } dim { size: 1 } } } }
    }
    # --- FLATTEN ---
    node {
      name: "flatten/Reshape"
      op: "Reshape"
      input: "conv1/Relu"
      attr { key: "dtype" value { type: DT_INT32 } }
      attr { key: "value" value { tensor { dtype: DT_INT32 tensor_shape { dim { size: 2 } } int_val: 0 int_val: 1959 } } }
      attr { key: "_output_shape" value { shape { dim { size: 50 } dim { size: 1960 } dim { size: 1 } dim { size: 1 } } } }
    }
    # --- LINEAR LAYER 2 ---
    node {
      name: "linear2/weight"
      op: "VariableV2"
      attr { key: "dtype" value { type: DT_FLOAT } }
      attr { key: "value" value { tensor { dtype: DT_FLOAT tensor_shape { dim { size: 100 } dim { size: 1960 } dim { size: 1 } dim { size: 1 } } float_val: 0.0 } } }
    }
    node {
      name: "linear2/bias"
      op: "BiasAdd"
      attr { key: "dtype" value { type: DT_FLOAT } }
      attr { key: "value" value { tensor { dtype: DT_FLOAT tensor_shape { dim { size: 100 } } float_val: 1.0 } } }
    }
    node {
      name: "linear2/MatMul"
      op: "MatMul"
      input: "flatten/Reshape"
      input: "linear2/weight"
      input: "linear2/bias"
      attr { key: "T" value { type: DT_FLOAT } }
      attr { key: "strides" value { list { i: 1 i: 1 i: 1 i: 1 } } }
      attr { key: "padding" value { s: "SAME" } }
      attr { key: "_output_shape" value { shape { dim { size: 50 } dim { size: 100 } dim { size: 1 } dim { size: 1 } } } }
    }
    node {
      name: "linear2/Relu"
      op: "Relu"
      input: "linear2/MatMul"
      attr { key: "shape" value { shape { dim { size: 10 } dim { size: 14 } dim { size: 14 } dim { size: 1 } } } }
      attr { key: "_output_shape" value { shape { dim { size: 50 } dim { size: 100 } dim { size: 1 } dim { size: 1 } } } }
    }
    # --- LINEAR LAYER 3 ---
    node {
      name: "linear3/weight"
      op: "VariableV2"
      attr { key: "dtype" value { type: DT_FLOAT } }
      attr { key: "value" value { tensor { dtype: DT_FLOAT tensor_shape { dim { size: 50 } dim { size: 100 } dim { size: 1 } dim { size: 1 } } float_val: 0.0 } } }
    }
    node {
      name: "linear3/bias"
      op: "BiasAdd"
      attr { key: "dtype" value { type: DT_FLOAT } }
      attr { key: "value" value { tensor { dtype: DT_FLOAT tensor_shape { dim { size: 10 } } float_val: 1.0 } } }
    }
    node {
      name: "linear3/MatMul"
      op: "MatMul"
      input: "linear2/Relu"
      input: "linear3/weight"
      input: "linear3/bias"
      attr { key: "T" value { type: DT_FLOAT } }
      attr { key: "strides" value { list { i: 1 i: 1 i: 1 i: 1 } } }
      attr { key: "padding" value { s: "SAME" } }
      attr { key: "_output_shape" value { shape { dim { size: 50 } dim { size: 10 } dim { size: 1 } dim { size: 1 } } } }
    }
    # --- SOFTMAX ---
    node {
      name: "output/weight"
      op: "Const"
      attr { key: "dtype" value { type: DT_FLOAT } }
      attr { key: "value" value { tensor { dtype: DT_FLOAT tensor_shape { dim { size: 10 } dim { size: 1 } } float_val: 0.01 } } }
    }
    node {
      name: "output/Softmax"
      op: "Softmax"
      input: "linear3/MatMul"
      input: "output/weight"
      attr { key: "_output_shape" value { shape { dim { size: 50 } dim { size: 10 } dim { size: 1 } dim { size: 1 } } } }
    }

    versions { producer: 440 }
    """

    # Parse and write using TF2 modern summary writer
    graph_def = graph_pb2.GraphDef()
    text_format.Parse(mnist_pbtxt, graph_def)

    # Use [tf.summary.create_file_writer](https://www.tensorflow.org)
    writer = tf.summary.create_file_writer(log_dir)
    with writer.as_default():
        tf.summary.graph(graph_def)
    
    print(f"✅ Success! View your graph at: tensorboard --logdir={log_dir}")

if __name__ == "__main__":
    create_mnist_cnn_graphdef()



