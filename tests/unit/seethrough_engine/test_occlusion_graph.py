import numpy as np

from seethrough_engine.occlusion_graph import compute_occlusion_graph

CANVAS = 128


def _edge(graph, front, back):
    return next(
        (e for e in graph["edges"] if e["front"] == front and e["back"] == back), None
    )


def _touching_layers():
    """`topwear` sits over the lower half of `neck`, with a soft low-alpha
    fringe along the seam so overlap (any presence) is larger than
    hidden_extent (fully opaque coverage) -- exactly what the plan's own
    JSON example shows (overlap_px 12850 > hidden_extent_px 4200)."""
    neck = np.zeros((CANVAS, CANVAS, 4), np.uint8)
    neck[20:80, 20:108, :3] = 200
    neck[20:80, 20:108, 3] = 255

    topwear = np.zeros((CANVAS, CANVAS, 4), np.uint8)
    topwear[60:120, 20:108, :3] = 180
    topwear[60:120, 20:108, 3] = 255
    topwear[60:64, 20:108, 3] = 60  # soft fringe: present, but not opaque

    return {"neck": neck, "topwear": topwear}


def test_touching_layers_produce_one_edge_with_front_and_back():
    graph = compute_occlusion_graph(_touching_layers())
    edge = _edge(graph, "topwear", "neck")
    assert edge is not None
    assert edge["overlap_px"] > edge["hidden_extent_px"] > 0
    assert edge["visible_boundary_px"] > 0
    assert 0.0 < edge["confidence"] <= 1.0
    assert 0.0 < edge["disocclusion_risk"] <= 1.0
    assert edge["depth_margin"] is None


def test_non_touching_layers_produce_no_edge():
    neck = np.zeros((CANVAS, CANVAS, 4), np.uint8)
    neck[0:20, 0:20, :3] = 200
    neck[0:20, 0:20, 3] = 255
    footwear = np.zeros((CANVAS, CANVAS, 4), np.uint8)
    footwear[100:120, 100:120, :3] = 100
    footwear[100:120, 100:120, 3] = 255

    graph = compute_occlusion_graph({"neck": neck, "footwear": footwear})
    assert graph["edges"] == []


def test_empty_layer_dict_returns_no_edges():
    graph = compute_occlusion_graph({})
    assert graph["edges"] == []
    assert graph["depth_available"] is False


def test_depth_dict_fills_in_depth_margin():
    layers = _touching_layers()
    depth = {
        "neck": np.full((CANVAS, CANVAS), 0.8, np.float32),      # further away
        "topwear": np.full((CANVAS, CANVAS), 0.2, np.float32),   # nearer
    }
    without_depth = compute_occlusion_graph(layers)
    with_depth = compute_occlusion_graph(layers, depth_dict=depth)

    edge = _edge(with_depth, "topwear", "neck")
    assert abs(edge["depth_margin"] - 0.6) < 1e-6
    assert with_depth["depth_available"] is True
    assert without_depth["depth_available"] is False
    assert _edge(without_depth, "topwear", "neck")["depth_margin"] is None


def test_edges_are_sorted_by_descending_risk():
    layers = _touching_layers()
    layers["legwear"] = np.zeros((CANVAS, CANVAS, 4), np.uint8)
    # A tiny, barely-touching pair: real edge, but far lower risk than the
    # broad neck/topwear contact above.
    layers["legwear"][78:82, 20:24, :3] = 90
    layers["legwear"][78:82, 20:24, 3] = 255

    graph = compute_occlusion_graph(layers)
    risks = [e["disocclusion_risk"] for e in graph["edges"]]
    assert risks == sorted(risks, reverse=True)
