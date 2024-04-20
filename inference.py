import torch
import numpy as np
from mask3d import get_model, load_mesh, prepare_data, map_output_to_pointcloud, save_colorized_mesh , CLASS_LABELS_200
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser.add_argument("--out_dir", type=str, help="out directory")
    parser_url.add_argument("--pointcloud_file", type=str, help="pointcloud_file")
    return parser

parser = get_args_parser()
args = parser.parse_args()

model = get_model('/scratch2/yuxili/checkpoints/scannet200_benchmark.ckpt')
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# load input data
pointcloud_file = args.pointcloud_file
out_dir = args.out_dir

mesh = load_mesh(out_dir + pointcloud_file + '.ply')

# prepare data
data, points, colors, features, unique_map, inverse_map = prepare_data(mesh, device)

# run model
with torch.no_grad():
    outputs = model(data, raw_coordinates=features)
    
# map output to point cloud
labels, instance_mapped = map_output_to_pointcloud(mesh, outputs, inverse_map)
print(np.unique(labels))
print(np.unique(instance_mapped))
for ele in np.array(np.unique(labels)):
    print(list(CLASS_LABELS_200)[int(ele)])
    
# save colorized mesh
save_colorized_mesh(mesh, labels, out_dir + pointcloud_file + '_labelled.ply', colormap='scannet200')
save_colorized_mesh(mesh, instance_mapped, out_dir  + pointcloud_file + '_instance.ply', colormap='scannet200')