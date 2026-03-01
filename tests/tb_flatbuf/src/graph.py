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
      attr { key: "shape" value { shape { dim { size: -1 } dim { size: 28 } dim { size: 28 } dim { size: 1 } } } }
    }

    # --- CONV LAYER 1 ---
    node {
      name: "conv1/weights"
      op: "Const"
      attr { key: "dtype" value { type: DT_FLOAT } }
      attr { key: "value" value { tensor { dtype: DT_FLOAT tensor_shape { dim { size: 3 } dim { size: 3 } dim { size: 1 } dim { size: 32 } } float_val: 0.1 } } }
    }
    node {
      name: "conv1/Conv2D"
      op: "Conv2D"
      input: "input"
      input: "conv1/weights"
      attr { key: "T" value { type: DT_FLOAT } }
      attr { key: "strides" value { list { i: 1 i: 1 i: 1 i: 1 } } }
      attr { key: "padding" value { s: "SAME" } }
    }
    node {
      name: "conv1/Relu"
      op: "Relu"
      input: "conv1/Conv2D"
    }
    node {
      name: "conv1/MaxPool"
      op: "MaxPool"
      input: "conv1/Relu"
      attr { key: "ksize" value { list { i: 1 i: 2 i: 2 i: 1 } } }
      attr { key: "strides" value { list { i: 1 i: 2 i: 2 i: 1 } } }
      attr { key: "padding" value { s: "SAME" } }
    }

    # --- CONV LAYER 2 ---
    node {
      name: "conv2/weights"
      op: "Const"
      attr { key: "dtype" value { type: DT_FLOAT } }
      attr { key: "value" value { tensor { dtype: DT_FLOAT tensor_shape { dim { size: 3 } dim { size: 3 } dim { size: 32 } dim { size: 64 } } float_val: 0.05 } } }
    }
    node {
      name: "conv2/Conv2D"
      op: "Conv2D"
      input: "conv1/MaxPool"
      input: "conv2/weights"
      attr { key: "T" value { type: DT_FLOAT } }
      attr { key: "strides" value { list { i: 1 i: 1 i: 1 i: 1 } } }
      attr { key: "padding" value { s: "SAME" } }
    }
    node {
      name: "conv2/Relu"
      op: "Relu"
      input: "conv2/Conv2D"
    }
    node {
      name: "conv2/MaxPool"
      op: "MaxPool"
      input: "conv2/Relu"
      attr { key: "ksize" value { list { i: 1 i: 2 i: 2 i: 1 } } }
      attr { key: "strides" value { list { i: 1 i: 2 i: 2 i: 1 } } }
      attr { key: "padding" value { s: "SAME" } }
    }

    # --- FLATTEN ---
    node {
      name: "flatten/shape"
      op: "Const"
      attr { key: "dtype" value { type: DT_INT32 } }
      attr { key: "value" value { tensor { dtype: DT_INT32 tensor_shape { dim { size: 2 } } int_val: -1 int_val: 3136 } } }
    }
    node {
      name: "flatten/Reshape"
      op: "Reshape"
      input: "conv2/MaxPool"
      input: "flatten/shape"
    }

    # --- DENSE & SOFTMAX ---
    node {
      name: "dense/weights"
      op: "Const"
      attr { key: "dtype" value { type: DT_FLOAT } }
      attr { key: "value" value { tensor { dtype: DT_FLOAT tensor_shape { dim { size: 3136 } dim { size: 10 } } float_val: 0.01 } } }
    }
    node {
      name: "dense/MatMul"
      op: "MatMul"
      input: "flatten/Reshape"
      input: "dense/weights"
    }
    node {
      name: "output/Softmax"
      op: "Softmax"
      input: "dense/MatMul"
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



