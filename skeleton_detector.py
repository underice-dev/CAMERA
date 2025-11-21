import numpy as np
from collections import deque
from skimage.morphology import skeletonize


def _neighbors(mask: np.ndarray, y: int, x: int):
    """Return 8-connected neighbor coordinates that are foreground."""
    h, w = mask.shape
    coords = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
                coords.append((ny, nx))
    return coords


def _prune_short_branches(skel: np.ndarray, min_branch_len: int = 8):
    """
    Iteratively remove short spurs to suppress noisy degree-1 branches.
    """
    pruned = skel.copy().astype(np.uint8)
    h, w = pruned.shape

    while True:
        endpoints = [
            (y, x)
            for y in range(h)
            for x in range(w)
            if pruned[y, x] and len(_neighbors(pruned, y, x)) == 1
        ]
        if not endpoints:
            break

        removed_any = False
        for ep in endpoints:
            path = [ep]
            prev = None
            cur = ep

            # Walk forward until hitting a junction/end
            while True:
                neigh = _neighbors(pruned, cur[0], cur[1])
                if prev is not None:
                    neigh = [n for n in neigh if n != prev]
                if len(neigh) != 1:
                    break

                prev = cur
                cur = neigh[0]
                path.append(cur)

                # Stop if we reach a junction or another endpoint
                if len(_neighbors(pruned, cur[0], cur[1])) != 2:
                    break

            if len(path) < min_branch_len:
                for py, px in path:
                    pruned[py, px] = 0
                removed_any = True

        if not removed_any:
            break

    return pruned


def _build_graph(skel: np.ndarray):
    """
    Build adjacency list for skeleton pixels.
    """
    coords = np.argwhere(skel > 0)
    if len(coords) == 0:
        return coords, {}, []

    idx_map = {tuple(p): i for i, p in enumerate(coords)}
    neighbors = [[] for _ in range(len(coords))]

    for i, (y, x) in enumerate(coords):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                ny, nx = y + dy, x + dx
                j = idx_map.get((ny, nx))
                if j is not None:
                    neighbors[i].append(j)

    return coords, idx_map, neighbors


def _bfs_farthest(start_idx: int, neighbors):
    dist = [-1] * len(neighbors)
    parent = [-1] * len(neighbors)
    dist[start_idx] = 0
    q = deque([start_idx])

    while q:
        cur = q.popleft()
        for nxt in neighbors[cur]:
            if dist[nxt] == -1:
                dist[nxt] = dist[cur] + 1
                parent[nxt] = cur
                q.append(nxt)

    far_idx = int(np.argmax(dist))
    return far_idx, dist, parent


def skeleton_longest_endpoints(mask: np.ndarray):
    """
    Locate the two farthest endpoints along the skeleton of a binary mask.

    Args:
        mask: uint8 binary image (0/255).

    Returns:
        dict with pt1, pt2 (x, y), length_px (geodesic on skeleton),
        and skeleton (uint8 image). Returns None if no valid skeleton.
    """
    if mask is None or mask.size == 0:
        return None

    skel = skeletonize(mask > 0)
    skel = _prune_short_branches(skel, min_branch_len=8)
    skel_u8 = (skel.astype(np.uint8)) * 255

    coords, idx_map, neighbors = _build_graph(skel)
    if len(coords) == 0 or len(neighbors) == 0:
        return None

    endpoints = [
        (y, x) for (y, x) in coords if len(_neighbors(skel, y, x)) == 1
    ]

    # Pick a robust start: endpoint if available, otherwise any pixel
    start_idx = idx_map[tuple(endpoints[0])] if endpoints else 0

    far_a, _, _ = _bfs_farthest(start_idx, neighbors)
    far_b, _, parent_b = _bfs_farthest(far_a, neighbors)

    # Reconstruct longest path from far_b back to far_a
    path_idx = []
    cur = far_b
    while cur != -1:
        path_idx.append(cur)
        cur = parent_b[cur]
    path_idx = path_idx[::-1]

    if len(path_idx) < 2:
        return None

    path_coords = coords[path_idx]
    diffs = np.diff(path_coords.astype(np.float32), axis=0)
    length_px = float(np.sum(np.linalg.norm(diffs, axis=1)))

    pt1_y, pt1_x = path_coords[0]
    pt2_y, pt2_x = path_coords[-1]

    return {
        "pt1": (int(pt1_x), int(pt1_y)),
        "pt2": (int(pt2_x), int(pt2_y)),
        "length_px": length_px,
        "skeleton": skel_u8
    }
