import hydra
import torch

from mask3d.models.mask3d import Mask3D
from mask3d.utils.utils import (
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)

class InstanceSegmentation(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = hydra.utils.instantiate(cfg.model)


    def forward(self, x, raw_coordinates=None):
        return self.model(x, raw_coordinates=raw_coordinates)
    

from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.experimental import initialize, compose

# imports for input loading
import albumentations as A
import MinkowskiEngine as ME
import numpy as np
import open3d as o3d

# # # imports for output
# from datasets.scannet200.scannet200_constants import (VALID_CLASS_IDS_20, VALID_CLASS_IDS_200, SCANNET_COLOR_MAP_20, SCANNET_COLOR_MAP_200)
### ScanNet Benchmark constants ###
VALID_CLASS_IDS_20 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)

CLASS_LABELS_20 = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                   'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                   'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')

SCANNET_COLOR_MAP_20 = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    14: (247., 182., 210.),
    15: (66., 188., 102.),
    16: (219., 219., 141.),
    17: (140., 57., 197.),
    18: (202., 185., 52.),
    19: (51., 176., 203.),
    20: (200., 54., 131.),
    21: (92., 193., 61.),
    22: (78., 71., 183.),
    23: (172., 114., 82.),
    24: (255., 127., 14.),
    25: (91., 163., 138.),
    26: (153., 98., 156.),
    27: (140., 153., 101.),
    28: (158., 218., 229.),
    29: (100., 125., 154.),
    30: (178., 127., 135.),
    32: (146., 111., 194.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    35: (96., 207., 209.),
    36: (227., 119., 194.),
    37: (213., 92., 176.),
    38: (94., 106., 211.),
    39: (82., 84., 163.),
    40: (100., 85., 144.),
}

### ScanNet200 Benchmark constants ###
VALID_CLASS_IDS_200 = (
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112, 115, 116, 118, 120, 121, 122, 125, 128, 130, 131, 132, 134, 136, 138, 139, 140, 141, 145, 148, 154,
155, 156, 157, 159, 161, 163, 165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193, 195, 202, 208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261, 264, 276, 283, 286, 300, 304, 312, 323, 325, 331, 342, 356, 370, 392, 395, 399, 408, 417,
488, 540, 562, 570, 572, 581, 609, 748, 776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191)

CLASS_LABELS_200 = (
'wall', 'chair', 'floor', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk', 'office chair', 'bed', 'pillow', 'sink', 'picture', 'window', 'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair', 'coffee table', 'box',
'refrigerator', 'lamp', 'kitchen cabinet', 'towel', 'clothes', 'tv', 'nightstand', 'counter', 'dresser', 'stool', 'cushion', 'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard', 'bag', 'backpack', 'toilet paper',
'printer', 'tv stand', 'whiteboard', 'blanket', 'shower curtain', 'trash can', 'closet', 'stairs', 'microwave', 'stove', 'shoe', 'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board', 'washing machine', 'mirror', 'copier',
'basket', 'sofa chair', 'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person', 'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard', 'piano', 'suitcase', 'rail', 'radiator', 'recycling bin', 'container',
'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock', 'stand', 'light', 'laundry basket', 'pipe', 'clothes dryer', 'guitar', 'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle', 'ladder', 'bathroom stall', 'shower wall',
'cup', 'jacket', 'storage bin', 'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat', 'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board', 'fireplace', 'soap dish', 'kitchen counter', 'doorframe',
'toilet paper dispenser', 'mini fridge', 'fire extinguisher', 'ball', 'hat', 'shower curtain rod', 'water cooler', 'paper cutter', 'tray', 'shower door', 'pillar', 'ledge', 'toaster oven', 'mouse', 'toilet seat cover dispenser',
'furniture', 'cart', 'storage container', 'scale', 'tissue box', 'light switch', 'crate', 'power outlet', 'decoration', 'sign', 'projector', 'closet door', 'vacuum cleaner', 'candle', 'plunger', 'stuffed animal', 'headphones', 'dish rack',
'broom', 'guitar case', 'range hood', 'dustpan', 'hair dryer', 'water bottle', 'handicap bar', 'purse', 'vent', 'shower floor', 'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock', 'music stand', 'projector screen', 'divider',
'laundry detergent', 'bathroom counter', 'object', 'bathroom vanity', 'closet wall', 'laundry hamper', 'bathroom stall door', 'ceiling light', 'trash bin', 'dumbbell', 'stair rail', 'tube', 'bathroom cabinet', 'cd case', 'closet rod',
'coffee kettle', 'structure', 'shower head', 'keyboard piano', 'case of water bottles', 'coat rack', 'storage organizer', 'folded chair', 'fire alarm', 'power strip', 'calendar', 'poster', 'potted plant', 'luggage', 'mattress')

SCANNET_COLOR_MAP_200 = {
0: (0., 0., 0.),
1: (174., 199., 232.),
2: (188., 189., 34.),
3: (152., 223., 138.),
4: (255., 152., 150.),
5: (214., 39., 40.),
6: (91., 135., 229.),
7: (31., 119., 180.),
8: (229., 91., 104.),
9: (247., 182., 210.),
10: (91., 229., 110.),
11: (255., 187., 120.),
13: (141., 91., 229.),
14: (112., 128., 144.),
15: (196., 156., 148.),
16: (197., 176., 213.),
17: (44., 160., 44.),
18: (148., 103., 189.),
19: (229., 91., 223.),
21: (219., 219., 141.),
22: (192., 229., 91.),
23: (88., 218., 137.),
24: (58., 98., 137.),
26: (177., 82., 239.),
27: (255., 127., 14.),
28: (237., 204., 37.),
29: (41., 206., 32.),
31: (62., 143., 148.),
32: (34., 14., 130.),
33: (143., 45., 115.),
34: (137., 63., 14.),
35: (23., 190., 207.),
36: (16., 212., 139.),
38: (90., 119., 201.),
39: (125., 30., 141.),
40: (150., 53., 56.),
41: (186., 197., 62.),
42: (227., 119., 194.),
44: (38., 100., 128.),
45: (120., 31., 243.),
46: (154., 59., 103.),
47: (169., 137., 78.),
48: (143., 245., 111.),
49: (37., 230., 205.),
50: (14., 16., 155.),
51: (196., 51., 182.),
52: (237., 80., 38.),
54: (138., 175., 62.),
55: (158., 218., 229.),
56: (38., 96., 167.),
57: (190., 77., 246.),
58: (208., 49., 84.),
59: (208., 193., 72.),
62: (55., 220., 57.),
63: (10., 125., 140.),
64: (76., 38., 202.),
65: (191., 28., 135.),
66: (211., 120., 42.),
67: (118., 174., 76.),
68: (17., 242., 171.),
69: (20., 65., 247.),
70: (208., 61., 222.),
71: (162., 62., 60.),
72: (210., 235., 62.),
73: (45., 152., 72.),
74: (35., 107., 149.),
75: (160., 89., 237.),
76: (227., 56., 125.),
77: (169., 143., 81.),
78: (42., 143., 20.),
79: (25., 160., 151.),
80: (82., 75., 227.),
82: (253., 59., 222.),
84: (240., 130., 89.),
86: (123., 172., 47.),
87: (71., 194., 133.),
88: (24., 94., 205.),
89: (134., 16., 179.),
90: (159., 32., 52.),
93: (213., 208., 88.),
95: (64., 158., 70.),
96: (18., 163., 194.),
97: (65., 29., 153.),
98: (177., 10., 109.),
99: (152., 83., 7.),
100: (83., 175., 30.),
101: (18., 199., 153.),
102: (61., 81., 208.),
103: (213., 85., 216.),
104: (170., 53., 42.),
105: (161., 192., 38.),
106: (23., 241., 91.),
107: (12., 103., 170.),
110: (151., 41., 245.),
112: (133., 51., 80.),
115: (184., 162., 91.),
116: (50., 138., 38.),
118: (31., 237., 236.),
120: (39., 19., 208.),
121: (223., 27., 180.),
122: (254., 141., 85.),
125: (97., 144., 39.),
128: (106., 231., 176.),
130: (12., 61., 162.),
131: (124., 66., 140.),
132: (137., 66., 73.),
134: (250., 253., 26.),
136: (55., 191., 73.),
138: (60., 126., 146.),
139: (153., 108., 234.),
140: (184., 58., 125.),
141: (135., 84., 14.),
145: (139., 248., 91.),
148: (53., 200., 172.),
154: (63., 69., 134.),
155: (190., 75., 186.),
156: (127., 63., 52.),
157: (141., 182., 25.),
159: (56., 144., 89.),
161: (64., 160., 250.),
163: (182., 86., 245.),
165: (139., 18., 53.),
166: (134., 120., 54.),
168: (49., 165., 42.),
169: (51., 128., 133.),
170: (44., 21., 163.),
177: (232., 93., 193.),
180: (176., 102., 54.),
185: (116., 217., 17.),
188: (54., 209., 150.),
191: (60., 99., 204.),
193: (129., 43., 144.),
195: (252., 100., 106.),
202: (187., 196., 73.),
208: (13., 158., 40.),
213: (52., 122., 152.),
214: (128., 76., 202.),
221: (187., 50., 115.),
229: (180., 141., 71.),
230: (77., 208., 35.),
232: (72., 183., 168.),
233: (97., 99., 203.),
242: (172., 22., 158.),
250: (155., 64., 40.),
261: (118., 159., 30.),
264: (69., 252., 148.),
276: (45., 103., 173.),
283: (111., 38., 149.),
286: (184., 9., 49.),
300: (188., 174., 67.),
304: (53., 206., 53.),
312: (97., 235., 252.),
323: (66., 32., 182.),
325: (236., 114., 195.),
331: (241., 154., 83.),
342: (133., 240., 52.),
356: (16., 205., 144.),
370: (75., 101., 198.),
392: (237., 95., 251.),
395: (191., 52., 49.),
399: (227., 254., 54.),
408: (49., 206., 87.),
417: (48., 113., 150.),
488: (125., 73., 182.),
540: (229., 32., 114.),
562: (158., 119., 28.),
570: (60., 205., 27.),
572: (18., 215., 201.),
581: (79., 76., 153.),
609: (134., 13., 116.),
748: (192., 97., 63.),
776: (108., 163., 18.),
1156: (95., 220., 156.),
1163: (98., 141., 208.),
1164: (144., 19., 193.),
1165: (166., 36., 57.),
1166: (212., 202., 34.),
1167: (23., 206., 34.),
1168: (91., 211., 236.),
1169: (79., 55., 137.),
1170: (182., 19., 117.),
1171: (134., 76., 14.),
1172: (87., 185., 28.),
1173: (82., 224., 187.),
1174: (92., 110., 214.),
1175: (168., 80., 171.),
1176: (197., 63., 51.),
1178: (175., 199., 77.),
1179: (62., 180., 98.),
1180: (8., 91., 150.),
1181: (77., 15., 130.),
1182: (154., 65., 96.),
1183: (197., 152., 11.),
1184: (59., 155., 45.),
1185: (12., 147., 145.),
1186: (54., 35., 219.),
1187: (210., 73., 181.),
1188: (221., 124., 77.),
1189: (149., 214., 66.),
1190: (72., 185., 134.),
1191: (42., 94., 198.),
}

### For instance segmentation the non-object categories ###
VALID_PANOPTIC_IDS = (1, 3)

CLASS_LABELS_PANOPTIC = ('wall', 'floor')

def get_model(checkpoint_path=None):


    # Initialize the directory with config files
    with initialize(config_path="conf"):
        # Compose a configuration
        cfg = compose(config_name="config_base_instance_segmentation.yaml")

    cfg.general.checkpoint = checkpoint_path

    # would be nicd to avoid this hardcoding below
    dataset_name = checkpoint_path.split('/')[-1].split('_')[0]
    if dataset_name == 'scannet200':
        cfg.general.num_targets = 201
        cfg.general.train_mode = False
        cfg.general.eval_on_segments = True
        cfg.general.topk_per_image = 300
        cfg.general.use_dbscan = True
        cfg.general.dbscan_eps = 0.95
        cfg.general.export_threshold = 0.001

        # # data
        cfg.data.num_labels = 200
        cfg.data.test_mode = "test"

        # # model
        cfg.model.num_queries = 150
        
    if dataset_name == 'scannet':
        cfg.general.num_targets = 19
        cfg.general.train_mode = False
        cfg.general.eval_on_segments = True
        cfg.general.topk_per_image = 300
        cfg.general.use_dbscan = True
        cfg.general.dbscan_eps = 0.95
        cfg.general.export_threshold = 0.001

        # # data
        cfg.data.num_labels = 20
        cfg.data.test_mode = "test"

        # # model
        cfg.model.num_queries = 150
        
        #TODO: this has to be fixed and discussed with Jonas
        # cfg.model.scene_min = -3.
        # cfg.model.scene_max = 3.

    # # Initialize the Hydra context
    # hydra.core.global_hydra.GlobalHydra.instance().clear()
    # hydra.initialize(config_path="conf")

    # Load the configuration
    # cfg = hydra.compose(config_name="config_base_instance_segmentation.yaml")
    model = InstanceSegmentation(cfg)

    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(
            cfg, model
        )
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    return model


def load_mesh(pcl_file):
    
    # load point cloud
    input_mesh_path = pcl_file
    mesh = o3d.io.read_triangle_mesh(input_mesh_path)
    return mesh

def prepare_data(mesh, device):
    
    # normalization for point cloud features
    color_mean = (0.47793125906962, 0.4303257521323044, 0.3749598901421883)
    color_std = (0.2834475483823543, 0.27566157565723015, 0.27018971370874995)
    normalize_color = A.Normalize(mean=color_mean, std=color_std)

    
    points = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    colors = colors * 255.

    pseudo_image = colors.astype(np.uint8)[np.newaxis, :, :]
    colors = np.squeeze(normalize_color(image=pseudo_image)["image"])

    coords = np.floor(points / 0.02)
    _, _, unique_map, inverse_map = ME.utils.sparse_quantize(
        coordinates=coords,
        features=colors,
        return_index=True,
        return_inverse=True,
    )

    sample_coordinates = coords[unique_map]
    coordinates = [torch.from_numpy(sample_coordinates).int()]
    sample_features = colors[unique_map]
    features = [torch.from_numpy(sample_features).float()]

    coordinates, _ = ME.utils.sparse_collate(coords=coordinates, feats=features)
    features = torch.cat(features, dim=0)
    data = ME.SparseTensor(
        coordinates=coordinates,
        features=features,
        device=device,
    )
    
    
    return data, points, colors, features, unique_map, inverse_map


def map_output_to_pointcloud(mesh, 
                             outputs, 
                             inverse_map, 
                             label_space='scannet200',
                             confidence_threshold=0.8):
    
    # parse predictions
    logits = outputs["pred_logits"]
    masks = outputs["pred_masks"]

    # reformat predictions
    logits = logits[0].detach().cpu()
    masks = masks[0].detach().cpu()
    
    labels = []
    instance = []
    confidences = []
    masks_binary = []

    for i in range(len(logits)):
        p_labels = torch.softmax(logits[i], dim=-1)
        p_masks = torch.sigmoid(masks[:, i])
        l = torch.argmax(p_labels, dim=-1)
        c_label = torch.max(p_labels)
        m = p_masks > 0.5
        c_m = p_masks[m].sum() / (m.sum() + 1e-8)
        c = c_label * c_m
        if l < 200 and c > confidence_threshold:
            instance.append(i)
            labels.append(l.item())
            confidences.append(c.item())
            masks_binary.append(
                m[inverse_map])  # mapping the mask back to the original point cloud
    
    # save labelled mesh
    mesh_labelled = o3d.geometry.TriangleMesh()
    mesh_labelled.vertices = mesh.vertices
    mesh_labelled.triangles = mesh.triangles

    labels_mapped = np.zeros((len(mesh.vertices), 1))
    instance_mapped = np.zeros((len(mesh.vertices), 1))

    for i, (l, c, m, inst) in enumerate(
        sorted(zip(labels, confidences, masks_binary, instance), reverse=False)):
        
        if label_space == 'scannet200':
            label_offset = 2
            if l == 0:
                l = -1 + label_offset
            else:
                l = int(l) + label_offset
                        
        labels_mapped[m == 1] = l
        instance_mapped[m == 1] = inst
        
    return labels_mapped, instance_mapped

def save_colorized_mesh(mesh, labels_mapped, output_file, colormap='scannet'):
    
    # colorize mesh
    colors = np.zeros((len(mesh.vertices), 3))
    for li in np.unique(labels_mapped):
        if colormap == 'scannet':
            raise ValueError('Not implemented yet')
        elif colormap == 'scannet200':
            v_li = VALID_CLASS_IDS_200[int(li)]
            colors[(labels_mapped == li)[:, 0], :] = SCANNET_COLOR_MAP_200[v_li]
        else:
            raise ValueError('Unknown colormap - not supported')
    
    colors = colors / 255.
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_triangle_mesh(output_file, mesh)

if __name__ == '__main__':
    
    model = get_model('checkpoints/scannet200/scannet200_benchmark.ckpt')
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # load input data
    pointcloud_file = 'data/pcl.ply'
    mesh = load_mesh(pointcloud_file)
    
    # prepare data
    data, points, colors, features, unique_map, inverse_map = prepare_data(mesh, device)
    
    # run model
    with torch.no_grad():
        outputs = model(data, raw_coordinates=features)
        
    # map output to point cloud
    labels = map_output_to_pointcloud(mesh, outputs, inverse_map)
    
    # save colorized mesh
    save_colorized_mesh(mesh, labels, 'data/pcl_labelled.ply', colormap='scannet200')
    