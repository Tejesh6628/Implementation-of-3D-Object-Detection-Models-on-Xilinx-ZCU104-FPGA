import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
    mlab = None
except Exception:
    try:
        import mayavi.mlab as mlab
        from visual_utils import visualize_utils as V
        OPEN3D_FLAG = False
    except Exception:
        class _DummyVis:
            @staticmethod
            def draw_scenes(*args, **kwargs):
                return None

        V = _DummyVis()
        mlab = None
        OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        self.logger = logger
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        if points.ndim != 2:
            points = points.reshape(-1, 4)

        if points.shape[0] < 1000:
            if self.logger is not None:
                self.logger.warning(
                    f'Only {points.shape[0]} points found in {self.sample_file_list[index]}; generating a denser point cloud for demo inference.'
                )

            point_cloud_range = getattr(self.dataset_cfg, 'POINT_CLOUD_RANGE', None)
            if point_cloud_range is not None and len(point_cloud_range) == 6:
                x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
            else:
                x_min, y_min, z_min, x_max, y_max, z_max = 0.0, -40.0, -3.0, 70.4, 40.0, 1.0

            rng = np.random.default_rng(0)
            num_points = max(1000, points.shape[0] * 20)
            generated_points = np.zeros((num_points, 4), dtype=np.float32)
            generated_points[:, 0] = rng.uniform(x_min, x_max, size=num_points)
            generated_points[:, 1] = rng.uniform(y_min, y_max, size=num_points)
            generated_points[:, 2] = rng.uniform(z_min, z_max, size=num_points)
            generated_points[:, 3] = 0.0

            if points.shape[0] > 0:
                generated_points[:points.shape[0]] = points[:points.shape[0]]

            points = generated_points

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)

    if args.ckpt:
        ckpt_path = Path(args.ckpt).expanduser()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f'Checkpoint file not found: {ckpt_path}')
        model.load_params_from_file(filename=str(ckpt_path), logger=logger, to_cpu=True)
    else:
        logger.warning('No checkpoint provided; using randomly initialized weights. Pass --ckpt to load a pretrained model.')

    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )

            if not OPEN3D_FLAG and mlab is not None:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
