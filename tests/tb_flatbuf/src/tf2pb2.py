from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorflow.python.framework import ops  # Often contains the graph types
from tensorflow.core.framework import graph_pb2 # The actual location of the proto
from tensorflow.python.framework import graph_io

event_file = '/tmp/tb_demo/graphs/tfevent.pb'

def convert_internal(event_file):
    for event in summary_iterator(event_file):
        if event.graph_def:
            # Load into the GraphDef object
            graph_def = graph_pb2.GraphDef()
            graph_def.ParseFromString(event.graph_def)
            
            # Write it out
            graph_io.write_graph(graph_def, './logs', 'graph.pbtxt', as_text=True)
            print("Done!")
            return




