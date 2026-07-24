import numpy as np

from repstudy.geometry import backproject
from repstudy.metrics import boundary_metrics, semantic_metrics
from repstudy.oracle import assign_oracle_labels, render_original_view
from repstudy.representations import build_representation
from repstudy.types import RGBDSample


def sample():
    rgb = np.zeros((8, 10, 3), dtype=np.uint8)
    depth = np.ones((8, 10), dtype=np.float32)
    labels = np.ones((8, 10), dtype=np.int32)
    labels[:, 5:] = 2
    labels[0, 0] = 0
    k = np.array([[10, 0, 4.0], [0, 10, 3.0], [0, 0, 1]], dtype=np.float32)
    return RGBDSample("synthetic", "synthetic", rgb, depth, labels, k)


def test_backprojection_coordinates():
    points, pixels = backproject(sample())
    center = np.where((pixels == [3, 4]).all(1))[0][0]
    assert np.allclose(points[center], [0, 0, 1])


def test_all_core_representations_render_and_label():
    item = sample()
    for name in ("pointcloud", "surfel", "mesh", "voxel", "tsdf", "superpoint"):
        result = assign_oracle_labels(build_representation(name, item), item)
        rendered = render_original_view(result, item)
        assert rendered.semantic_map.shape == item.semantic_gt.shape
        assert rendered.coverage_mask.any()
        assert result.attributes["oracle_label"].shape[0] == result.element_count


def test_tracks_and_boundary_metrics_are_separate():
    item = sample(); prediction = item.semantic_gt.copy(); prediction[:, 0] = 0
    coverage = prediction > 0
    values = semantic_metrics(prediction, item.semantic_gt, coverage, item.valid_label_mask, item.valid_depth_mask)
    assert values["valid_depth"]["coverage"] < 1
    assert values["full_label"]["missing_rate"] > 0
    assert "1" in boundary_metrics(prediction, item.semantic_gt)
