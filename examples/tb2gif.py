#
# @file
# @brief - Tensorboard event images to dynamic GIF extractor
#
import io
import argparse
import os
import shutil

import imageio.v2 as imageio
import matplotlib as mpl
import numpy as np

from PIL import Image
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.compat.proto import tensor_pb2

def get_args():
    parser = argparse.ArgumentParser(description='Create a gif from tensorboard file images')
    parser.add_argument('filename', type=str, help='Path to tensorboard file')
    parser.add_argument('tag',      type=str, help='Name of the image in the tensorboard e.g. `test/image`.')
    parser.add_argument('--check',  type=int, default=0)
    parser.add_argument('--output', default="./tb2gif_out.gif", type=str, help='File to store the final result')
    parser.add_argument('--start',  type=int, default=-1)
    parser.add_argument('--stop',   type=int, default=np.inf)
    
    return parser.parse_args()

def cleanup(path):
    if   os.path.isfile(path):  os.remove(path)
    elif os.path.isdir(path):   shutil.rmtree(path)
    else:
        raise ValueError("file {} is not a file or dir.".format(path))

def decode_tensor_image(tensor_proto):
    tensor = tensor_pb2.TensorProto()
    tensor.CopyFrom(tensor_proto)
    encoded_png = tensor.string_val[2]
    return Image.open(io.BytesIO(encoded_png))

def check_event(fn, start=-1, stop=np.inf):
    loader = event_file_loader.LegacyEventFileLoader(fn)

    for e in loader.Load():
        if not hasattr(e, 'summary'): continue
        if e.step < start: continue
        if e.step > stop:  break

        for e in loader.Load():
            if not hasattr(e, 'summary'): continue
            for v in e.summary.value:
                kind  = v.WhichOneof('value')
                dtype = v.metadata.plugin_data.plugin_name
                if (kind != 'tensor' or dtype != 'images'): continue
                print(f"step={e.step}, tag={v.tag!r}")

def process_event(fn, tag, start=-1, stop=np.inf, output_dir="./"):
    assert os.path.isdir(output_dir)

    names = []
    loader = event_file_loader.LegacyEventFileLoader(fn)

    for e in loader.Load():
        if not hasattr(e, 'summary'): continue
        if e.step < start: continue
        if e.step > stop:  break

        for v in e.summary.value:
            if v.tag != tag: continue
            
            im = decode_tensor_image(v.tensor)
            output_fn = os.path.realpath(f'{output_dir}/im_{e.step:06d}.png')
            print("processing ", output_fn)
            im.save(output_fn)
            
            names.append(output_fn)
            
    return names


def image_list_to_gif(filenames, output_fn):
    with imageio.get_writer(output_fn, mode='I', duration=0.5) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)


if __name__ == "__main__":
    args = get_args()
    tmp_output_dir = "/tmp/tb2gif/"

    if not os.path.isdir(tmp_output_dir):
        os.mkdir(tmp_output_dir)

    if (args.check):
        check_event(args.filename, start = args.start, stop = args.stop)
    else:
        names = process_event(
            args.filename,
            args.tag,
            start = args.start,
            stop  = args.stop,
            output_dir=tmp_output_dir)

        image_list_to_gif(names, args.output)
        cleanup(tmp_output_dir)
    
        print("gif created:", args.output)
