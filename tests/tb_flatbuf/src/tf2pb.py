import tensorflow as tf
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2

# Replace with the path to your tfevents file
input_path = './logs/tfevent.pb'
output_path = './logs/graph.pbtxt'

def export_graph_from_tfevents(event_file, output_file):
    # Initialize a GraphDef object
    graph_def = graph_pb2.GraphDef()
    graph_found = False

    # Iterate through the event file
    for event in tf.compat.v1.train.summary_iterator(event_file):
        if event.graph_def:
            graph_def.ParseFromString(event.graph_def)
            graph_found = True
            break # Stop after finding the first graph

    if graph_found:
        # Write as text-based .pbtxt
        with open(output_file, 'w') as f:
            f.write(text_format.MessageToString(graph_def))
        print(f"Success! Graph saved to {output_file}")
    else:
        print("No graph definition found in the provided tfevents file.")

export_graph_from_tfevents(input_path, output_path)
