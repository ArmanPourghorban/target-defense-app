import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from matplotlib.patches import Circle


# ============================================================
# Image / boundary utilities
# ============================================================
def extract_edge_points_from_uploaded_image(uploaded_file, num_samples=300, threshold=127, invert=True):
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, threshold, 255, thresh_type)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None, None, image_rgb, binary, image_rgb.shape[0]

    contour = max(contours, key=cv2.contourArea)
    contour = contour[:, 0, :].astype(float)

    diffs = np.diff(contour, axis=0, append=contour[:1])
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.cumsum(seg_lengths)
    total_length = cumulative[-1]

    if total_length <= 1e-12:
        return None, None, image_rgb, binary, image_rgb.shape[0]

    sample_distances = np.linspace(0, total_length, num_samples, endpoint=False)

    sampled_points = []
    j = 0
    for d in sample_distances:
        while j < len(cumulative) - 1 and cumulative[j] < d:
            j += 1
        sampled_points.append(contour[j])

    return np.array(sampled_points, dtype=float), contour, image_rgb, binary, image_rgb.shape[0]


def image_to_cartesian(points, image_height):
    pts = np.asarray(points, dtype=float).copy()
    pts[:, 1] = image_height - pts[:, 1]
    return pts


# ============================================================
# Boundary preprocessing
# ============================================================
def make_boundary_data(boundary_points):
    bp = np.asarray(boundary_points, dtype=float)

    if not np.allclose(bp[0], bp[-1]):
        closed = np.vstack([bp, bp[0]])
    else:
        closed = bp.copy()

    seg_a = closed[:-1]
    seg_b = closed[1:]
    seg_ab = seg_b - seg_a
    seg_len2 = np.sum(seg_ab * seg_ab, axis=1)
    seg_len2[seg_len2 < 1e-15] = 1e-15

    return {
        "points": bp,
        "closed": closed,
        "seg_a": seg_a,
        "seg_b": seg_b,
        "seg_ab": seg_ab,
        "seg_len2": seg_len2
    }


def point_to_segments_min_distance(point, boundary_data):
    p = np.asarray(point, dtype=float)

    a = boundary_data["seg_a"]
    ab = boundary_data["seg_ab"]
    len2 = boundary_data["seg_len2"]

    ap = p - a
    t = np.sum(ap * ab, axis=1) / len2
    t = np.clip(t, 0.0, 1.0)

    closest = a + t[:, None] * ab
    dists = np.linalg.norm(closest - p, axis=1)

    idx = int(np.argmin(dists))
    return closest[idx], float(dists[idx]), idx


def points_to_polygon_distances(points, boundary_data):
    pts = np.asarray(points, dtype=float)

    a = boundary_data["seg_a"]
    ab = boundary_data["seg_ab"]
    len2 = boundary_data["seg_len2"]

    ap = pts[:, None, :] - a[None, :, :]
    t = np.sum(ap * ab[None, :, :], axis=2) / len2[None, :]
    t = np.clip(t, 0.0, 1.0)

    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    dists = np.linalg.norm(closest - pts[:, None, :], axis=2)

    return np.min(dists, axis=1)


def closest_point_on_polygon(point, boundary_data):
    closest, dist, idx = point_to_segments_min_distance(point, boundary_data)
    return closest, dist, idx


def is_point_inside_polygon(point, polygon):
    x, y = point
    poly = np.asarray(polygon, dtype=float)

    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])

    inside = False
    for i in range(len(poly) - 1):
        x1, y1 = poly[i]
        x2, y2 = poly[i + 1]

        cond = ((y1 > y) != (y2 > y))
        if cond:
            xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1
            if x < xinters:
                inside = not inside

    return inside


def polygon_centroid(points):
    pts = np.asarray(points, dtype=float)

    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    x = pts[:, 0]
    y = pts[:, 1]

    a = 0.0
    cx = 0.0
    cy = 0.0

    for i in range(len(pts) - 1):
        cross = x[i] * y[i + 1] - x[i + 1] * y[i]
        a += cross
        cx += (x[i] + x[i + 1]) * cross
        cy += (y[i] + y[i + 1]) * cross

    a *= 0.5

    if abs(a) < 1e-12:
        return np.mean(points, axis=0)

    return np.array([cx / (6 * a), cy / (6 * a)])


# ============================================================
# Motion utilities
# ============================================================
def move_toward(current_pos, target_pos, speed, dt):
    current_pos = np.asarray(current_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)

    vec = target_pos - current_pos
    d = np.linalg.norm(vec)

    if d < 1e-12:
        return target_pos.copy(), True

    step = speed * dt

    if step >= d:
        return target_pos.copy(), True

    return current_pos + (step / d) * vec, False


def point_along_segment_by_time(start, end, speed, t):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    vec = end - start
    d = np.linalg.norm(vec)

    if d < 1e-12:
        return end.copy()

    travel = speed * t

    if travel >= d:
        return end.copy()

    return start + (travel / d) * vec


def circle_from_center_radius(center, radius, n_plot=80):
    theta = np.linspace(0, 2 * np.pi, n_plot)
    return np.column_stack([
        center[0] + radius * np.cos(theta),
        center[1] + radius * np.sin(theta)
    ])


def sample_circle_points(center, radius, n_points=16):
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    return np.column_stack([
        center[0] + radius * np.cos(theta),
        center[1] + radius * np.sin(theta)
    ])


# ============================================================
# Drone drawing
# ============================================================
def draw_drone(ax, pos, color="blue", size=4.0, alpha=1.0):
    x, y = pos
    s = size

    ax.plot([x - s, x + s], [y, y], color=color, linewidth=2, alpha=alpha)
    ax.plot([x, x], [y - s, y + s], color=color, linewidth=2, alpha=alpha)

    body = Circle(
        (x, y),
        radius=0.28 * s,
        facecolor=color,
        edgecolor="black",
        linewidth=0.8,
        alpha=alpha
    )
    ax.add_patch(body)

    rotor_r = 0.22 * s
    rotors = [
        (x - s, y),
        (x + s, y),
        (x, y - s),
        (x, y + s)
    ]

    for rx, ry in rotors:
        rotor = Circle(
            (rx, ry),
            radius=rotor_r,
            facecolor="white",
            edgecolor=color,
            linewidth=1.4,
            alpha=alpha
        )
        ax.add_patch(rotor)


# ============================================================
# Apollonius functions
# ============================================================
def apollonius_circle(attacker_pos, sensing_point, nu):
    A = np.asarray(attacker_pos, dtype=float)
    S = np.asarray(sensing_point, dtype=float)

    denom = 1.0 - nu**2

    if abs(denom) < 1e-12:
        raise ValueError("nu is too close to 1. Choose nu < 1.")

    center = (A - (nu**2) * S) / denom
    radius = (nu * np.linalg.norm(A - S)) / denom

    return center, radius


def circle_intersects_polygon_fast(center, radius, boundary_data, tol=1e-6):
    _, dmin, _ = point_to_segments_min_distance(center, boundary_data)
    return dmin <= radius + tol


def apollonius_circle_worst_distance_fast(center, radius, boundary_data, n_circle_samples=60):
    circle_pts = sample_circle_points(center, radius, n_points=n_circle_samples)
    dists = points_to_polygon_distances(circle_pts, boundary_data)

    max_idx = int(np.argmax(dists))
    worst_point = circle_pts[max_idx]
    max_dist = float(dists[max_idx])

    return max_dist, worst_point


def defender_hidden_until_engagement(
    defender_start,
    engagement_point,
    attacker_start_now,
    attacker_sample_point,
    rhoA,
    defender_speed,
    attacker_speed,
    engagement_time_from_now,
    n_samples=10
):
    td = np.linalg.norm(np.asarray(defender_start) - np.asarray(engagement_point)) / defender_speed
    ta = engagement_time_from_now

    if td >= ta:
        return False

    times = np.linspace(0.0, ta, n_samples, endpoint=False)

    for t in times:
        if t < td:
            D_t = point_along_segment_by_time(defender_start, engagement_point, defender_speed, t)
        else:
            D_t = np.asarray(engagement_point, dtype=float)

        A_t = point_along_segment_by_time(attacker_start_now, attacker_sample_point, attacker_speed, t)

        if np.linalg.norm(D_t - A_t) <= rhoA:
            return False

    return True



# ============================================================
# Pure pursuit helper functions
# ============================================================
def furthest_point_on_circle_from_point(center, radius, reference_point):
    center = np.asarray(center, dtype=float)
    reference_point = np.asarray(reference_point, dtype=float)

    direction = center - reference_point
    norm_direction = np.linalg.norm(direction)

    if norm_direction < 1e-12:
        direction = np.array([1.0, 0.0])
    else:
        direction = direction / norm_direction

    return center + radius * direction


def circle_polygon_intersection_points(center, radius, boundary_data, tol=1e-6):
    center = np.asarray(center, dtype=float)
    intersections = []

    seg_a = boundary_data["seg_a"]
    seg_b = boundary_data["seg_b"]

    for a, b in zip(seg_a, seg_b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        d = b - a
        f = a - center

        A = np.dot(d, d)
        if A < 1e-15:
            continue

        B = 2 * np.dot(f, d)
        C = np.dot(f, f) - radius**2
        disc = B**2 - 4 * A * C

        if disc < -tol:
            continue

        disc = max(disc, 0.0)
        sqrt_disc = np.sqrt(disc)

        t1 = (-B - sqrt_disc) / (2 * A)
        t2 = (-B + sqrt_disc) / (2 * A)

        for t in [t1, t2]:
            if -tol <= t <= 1.0 + tol:
                p = a + np.clip(t, 0.0, 1.0) * d
                intersections.append(p)

    unique = []
    for p in intersections:
        if not any(np.linalg.norm(p - q) < 1e-4 for q in unique):
            unique.append(p)

    return unique


def choose_pure_pursuit_capture_point(center, radius, boundary_data, centroid, attacker_pos):
    intersections = circle_polygon_intersection_points(center, radius, boundary_data)

    if len(intersections) > 0:
        dists = [np.linalg.norm(p - attacker_pos) for p in intersections]
        return intersections[int(np.argmin(dists))], True

    return furthest_point_on_circle_from_point(center, radius, centroid), False

# ============================================================
# Spawn geometry modes
# ============================================================
def generate_attackers_random_outside_box(boundary_points, n_attackers, offset_margin=40.0, seed=0):
    rng = np.random.default_rng(seed)

    boundary_points = np.asarray(boundary_points, dtype=float)

    xmin, ymin = np.min(boundary_points, axis=0)
    xmax, ymax = np.max(boundary_points, axis=0)

    xmin -= offset_margin
    xmax += offset_margin
    ymin -= offset_margin
    ymax += offset_margin

    attackers = []
    max_trials = 50000
    trials = 0

    while len(attackers) < n_attackers and trials < max_trials:
        p = np.array([
            rng.uniform(xmin, xmax),
            rng.uniform(ymin, ymax)
        ], dtype=float)

        if not is_point_inside_polygon(p, boundary_points):
            attackers.append(p)

        trials += 1

    return np.array(attackers), None


def generate_attackers_on_big_circle(boundary_points, n_attackers, radius_offset=40.0, seed=0):
    rng = np.random.default_rng(seed)

    boundary_points = np.asarray(boundary_points, dtype=float)
    centroid = polygon_centroid(boundary_points)

    distances = np.linalg.norm(boundary_points - centroid, axis=1)
    base_radius = np.max(distances)
    R = base_radius + radius_offset

    attackers = []

    for _ in range(n_attackers):
        theta = rng.uniform(0, 2 * np.pi)
        p = centroid + R * np.array([np.cos(theta), np.sin(theta)])
        attackers.append(p)

    circle_preview = circle_from_center_radius(centroid, R, n_plot=240)

    return np.array(attackers), circle_preview


def generate_attackers_on_scaled_boundary(boundary_points, n_attackers, scale_factor=1.25, seed=0):
    rng = np.random.default_rng(seed)

    boundary_points = np.asarray(boundary_points, dtype=float)
    centroid = polygon_centroid(boundary_points)

    scaled_boundary = centroid + scale_factor * (boundary_points - centroid)

    idx = rng.integers(0, len(scaled_boundary), size=n_attackers)
    attackers = scaled_boundary[idx]

    return np.array(attackers), scaled_boundary


# ============================================================
# Attackers and defenders
# ============================================================
def initialize_attackers(boundary_data, starts):
    attackers = []

    for i, start in enumerate(starts):
        closest_pt, _, _ = closest_point_on_polygon(start, boundary_data)

        attackers.append({
            "id": i,
            "start": np.array(start, dtype=float),
            "pos": np.array(start, dtype=float),
            "closest_point": np.array(closest_pt, dtype=float),
            "active": False,
            "captured": False,
            "breached": False,
            "eliminated": False,
            "spawned": False,
            "assigned_defender_id": None
        })

    return attackers


def initialize_defenders(centroid, n_defenders, spacing=5.0):
    defenders = []
    centroid = np.asarray(centroid, dtype=float)

    if n_defenders == 1:
        positions = [centroid]
    else:
        angles = np.linspace(0, 2 * np.pi, n_defenders, endpoint=False)
        positions = [
            centroid + spacing * np.array([np.cos(a), np.sin(a)])
            for a in angles
        ]

    for i, pos in enumerate(positions):
        defenders.append({
            "id": i,
            "pos": np.array(pos, dtype=float),
            "mode": "seek",
            "target_id": None,
            "plan": None
        })

    return defenders


def release_defender_from_attacker(defenders, attacker_id):
    for defender in defenders:
        if defender["target_id"] == attacker_id:
            defender["target_id"] = None
            defender["plan"] = None
            defender["mode"] = "seek"


# ============================================================
# Optimized per-defender Apollonius policy
# ============================================================
def evaluate_attacker_for_defender_fast(
    attacker,
    defender_pos,
    boundary_data,
    nu,
    rhoA,
    n_future_samples=4,
    n_sensing_samples=8,
    n_circle_samples=60,
    defender_speed=1.0
):
    if (
        attacker["captured"]
        or attacker["breached"]
        or attacker["eliminated"]
        or not attacker["active"]
    ):
        return None

    A_now = np.array(attacker["pos"], dtype=float)
    target_pt = np.array(attacker["closest_point"], dtype=float)

    remaining_dist = np.linalg.norm(target_pt - A_now)

    if remaining_dist < 1e-10:
        return None

    remaining_time = remaining_dist / nu
    future_times = np.linspace(0.0, remaining_time, n_future_samples)

    best_score = np.inf
    best_data = None

    for tau in future_times[:-1]:
        if tau <= 1e-12:
            continue

        A_sample = point_along_segment_by_time(A_now, target_pt, nu, tau)
        sensing_pts = sample_circle_points(A_sample, rhoA, n_points=n_sensing_samples)

        defender_times = np.linalg.norm(sensing_pts - defender_pos, axis=1) / defender_speed
        feasible_idx = np.where(defender_times < tau)[0]

        for idx in feasible_idx:
            S = sensing_pts[idx]

            hidden_ok = defender_hidden_until_engagement(
                defender_start=defender_pos,
                engagement_point=S,
                attacker_start_now=A_now,
                attacker_sample_point=A_sample,
                rhoA=rhoA,
                defender_speed=defender_speed,
                attacker_speed=nu,
                engagement_time_from_now=tau,
                n_samples=10
            )

            if not hidden_ok:
                continue

            center, radius = apollonius_circle(A_sample, S, nu)

            if circle_intersects_polygon_fast(center, radius, boundary_data):
                continue

            score, worst_point = apollonius_circle_worst_distance_fast(
                center=center,
                radius=radius,
                boundary_data=boundary_data,
                n_circle_samples=n_circle_samples
            )

            capture_leg = np.linalg.norm(S - worst_point) / defender_speed
            total_capture_time = tau + capture_leg

            if score < best_score:
                best_score = score

                best_data = {
                    "attacker_id": attacker["id"],
                    "attacker_sample_point": A_sample.copy(),
                    "engagement_point": np.array(S, dtype=float),
                    "capture_point": np.array(worst_point, dtype=float),
                    "score": float(score),
                    "attacker_sample_time_rel": float(tau),
                    "capture_finish_time_rel": float(total_capture_time)
                }

    return best_data


def choose_best_attacker_for_defender_fast(
    defender,
    attackers,
    boundary_data,
    nu,
    rhoA,
    n_future_samples=4,
    n_sensing_samples=8,
    n_circle_samples=60,
    defender_speed=1.0,
    max_candidates=8
):
    active_candidates = [
        a for a in attackers
        if a["active"]
        and not a["captured"]
        and not a["breached"]
        and not a["eliminated"]
        and a["assigned_defender_id"] is None
    ]

    if len(active_candidates) == 0:
        return None

    dists = np.array([
        np.linalg.norm(defender["pos"] - a["pos"])
        for a in active_candidates
    ])

    order = np.argsort(dists)
    active_candidates = [active_candidates[i] for i in order[:max_candidates]]

    results = []

    for attacker in active_candidates:
        res = evaluate_attacker_for_defender_fast(
            attacker=attacker,
            defender_pos=defender["pos"],
            boundary_data=boundary_data,
            nu=nu,
            rhoA=rhoA,
            n_future_samples=n_future_samples,
            n_sensing_samples=n_sensing_samples,
            n_circle_samples=n_circle_samples,
            defender_speed=defender_speed
        )

        if res is not None:
            results.append(res)

    if len(results) == 0:
        return None

    return min(results, key=lambda r: r["capture_finish_time_rel"])


def assign_free_defenders_apollonius_policy_fast(
    defenders,
    attackers,
    boundary_data,
    nu,
    rhoA,
    n_future_samples,
    n_sensing_samples,
    n_circle_samples,
    defender_speed,
    max_candidates
):
    free_defenders = [
        d for d in defenders
        if d["target_id"] is None and d["mode"] in ["seek", "move_to_centroid"]
    ]

    for defender in free_defenders:
        best = choose_best_attacker_for_defender_fast(
            defender=defender,
            attackers=attackers,
            boundary_data=boundary_data,
            nu=nu,
            rhoA=rhoA,
            n_future_samples=n_future_samples,
            n_sensing_samples=n_sensing_samples,
            n_circle_samples=n_circle_samples,
            defender_speed=defender_speed,
            max_candidates=max_candidates
        )

        if best is not None:
            defender["plan"] = best
            defender["target_id"] = best["attacker_id"]
            defender["mode"] = "move_to_engagement"

            target = next((a for a in attackers if a["id"] == best["attacker_id"]), None)
            if target is not None:
                target["assigned_defender_id"] = defender["id"]


def update_defenders_apollonius_game(
    defenders,
    attackers,
    centroid,
    defender_speed,
    attacker_speed,
    dt
):
    for defender in defenders:
        mode = defender["mode"]
        plan = defender["plan"]

        if defender["target_id"] is not None:
            target = next((a for a in attackers if a["id"] == defender["target_id"]), None)

            if (
                target is None
                or target["captured"]
                or target["breached"]
                or target["eliminated"]
                or not target["active"]
            ):
                defender["target_id"] = None
                defender["plan"] = None
                defender["mode"] = "seek"
                continue

        if mode == "move_to_engagement" and plan is not None:
            defender["pos"], arrived = move_toward(
                defender["pos"],
                plan["engagement_point"],
                defender_speed,
                dt
            )

            plan["attacker_sample_time_rel"] = max(0.0, plan["attacker_sample_time_rel"] - dt)
            plan["capture_finish_time_rel"] = max(0.0, plan["capture_finish_time_rel"] - dt)

            if arrived:
                defender["mode"] = "wait_for_attacker"

        elif mode == "wait_for_attacker" and plan is not None:
            plan["attacker_sample_time_rel"] = max(0.0, plan["attacker_sample_time_rel"] - dt)
            plan["capture_finish_time_rel"] = max(0.0, plan["capture_finish_time_rel"] - dt)

            if plan["attacker_sample_time_rel"] <= 0:
                defender["mode"] = "move_to_capture"

        elif mode == "move_to_capture" and plan is not None:
            target = next((a for a in attackers if a["id"] == defender["target_id"]), None)

            defender["pos"], arrived_def = move_toward(
                defender["pos"],
                plan["capture_point"],
                defender_speed,
                dt
            )

            arrived_att = False

            if target is not None and target["active"] and not target["captured"] and not target["breached"]:
                target["pos"], arrived_att = move_toward(
                    target["pos"],
                    plan["capture_point"],
                    attacker_speed,
                    dt
                )

            if target is not None:
                if np.linalg.norm(defender["pos"] - target["pos"]) < 1e-2 or arrived_def or arrived_att:
                    target["captured"] = True
                    target["active"] = False
                    target["eliminated"] = True
                    target["assigned_defender_id"] = None
                    target["pos"] = plan["capture_point"].copy()

                    defender["target_id"] = None
                    defender["plan"] = None
                    defender["mode"] = "seek"

        elif mode == "move_to_centroid":
            defender["pos"], arrived = move_toward(
                defender["pos"],
                centroid,
                defender_speed,
                dt
            )

            if arrived:
                defender["mode"] = "seek"

        elif mode == "seek":
            defender["mode"] = "move_to_centroid"



# ============================================================
# Pure pursuit defender policy
# ============================================================
def release_defender_only(defender):
    defender["target_id"] = None
    defender["plan"] = None
    defender["mode"] = "seek"


def release_pure_pursuit_defender_from_attacker(defenders, attackers, attacker_id):
    target = next((a for a in attackers if a["id"] == attacker_id), None)
    if target is not None:
        target["assigned_defender_id"] = None

    for defender in defenders:
        if defender["target_id"] == attacker_id:
            release_defender_only(defender)


def assign_pure_pursuit_defenders(defenders, attackers):
    free_defenders = [
        d for d in defenders
        if d["target_id"] is None and d["mode"] in ["seek", "move_to_centroid"]
    ]

    active_candidates = [
        a for a in attackers
        if a["active"]
        and not a["captured"]
        and not a["breached"]
        and not a["eliminated"]
        and a["assigned_defender_id"] is None
    ]

    for defender in free_defenders:
        if len(active_candidates) == 0:
            break

        dists = np.array([np.linalg.norm(defender["pos"] - attacker["pos"]) for attacker in active_candidates])
        best_idx = int(np.argmin(dists))
        chosen = active_candidates.pop(best_idx)

        defender["target_id"] = chosen["id"]
        defender["mode"] = "pure_pursuit"
        defender["plan"] = None
        chosen["assigned_defender_id"] = defender["id"]


def update_defenders_pure_pursuit(defenders, attackers, centroid, boundary_data, nu, rhoA, defender_speed, dt):
    for defender in defenders:
        if defender["target_id"] is not None:
            target = next((a for a in attackers if a["id"] == defender["target_id"]), None)
            if target is None or target["captured"] or target["breached"] or target["eliminated"] or not target["active"]:
                old_target_id = defender["target_id"]
                release_pure_pursuit_defender_from_attacker(defenders, attackers, old_target_id)
                continue

        if defender["mode"] == "pure_pursuit":
            target = next((a for a in attackers if a["id"] == defender["target_id"]), None)
            if target is None:
                release_defender_only(defender)
                continue

            defender["pos"], _ = move_toward(defender["pos"], target["pos"], defender_speed, dt)

            if np.linalg.norm(defender["pos"] - target["pos"]) <= rhoA:
                center, radius = apollonius_circle(target["pos"], defender["pos"], nu)
                capture_point, intersects_boundary = choose_pure_pursuit_capture_point(
                    center, radius, boundary_data, centroid, target["pos"]
                )

                defender["plan"] = {
                    "attacker_id": target["id"],
                    "capture_point": capture_point,
                    "intersects_boundary": intersects_boundary
                }
                defender["mode"] = "pure_pursuit_capture"

        elif defender["mode"] == "pure_pursuit_capture":
            target = next((a for a in attackers if a["id"] == defender["target_id"]), None)
            plan = defender["plan"]
            if target is None or plan is None:
                release_defender_only(defender)
                continue

            capture_point = plan["capture_point"]

            target["pos"], arrived_att = move_toward(target["pos"], capture_point, nu, dt)
            defender["pos"], arrived_def = move_toward(defender["pos"], target["pos"], defender_speed, dt)

            if np.linalg.norm(defender["pos"] - target["pos"]) < 1e-2 or arrived_def:
                target["captured"] = True
                target["active"] = False
                target["eliminated"] = True
                target["pos"] = defender["pos"].copy()
                target["assigned_defender_id"] = None
                release_defender_only(defender)

            elif arrived_att:
                target["breached"] = True
                target["active"] = False
                target["eliminated"] = True
                target["assigned_defender_id"] = None
                release_defender_only(defender)

        elif defender["mode"] == "move_to_centroid":
            defender["pos"], arrived = move_toward(defender["pos"], centroid, defender_speed, dt)
            if arrived:
                defender["mode"] = "seek"

        elif defender["mode"] == "seek":
            defender["mode"] = "move_to_centroid"

# ============================================================
# Plotting
# ============================================================
def draw_game(
    boundary_data,
    defenders,
    attackers,
    centroid,
    rhoA,
    drone_size,
    current_time=0.0,
    extra_curve=None,
    fixed_xlim=None,
    fixed_ylim=None
):
    fig, ax = plt.subplots(figsize=(8, 8))

    boundary_closed = boundary_data["closed"]
    ax.plot(boundary_closed[:, 0], boundary_closed[:, 1], 'k-', linewidth=2.5, label="Boundary")

    if extra_curve is not None:
        extra_curve = np.asarray(extra_curve, dtype=float)

        if not np.allclose(extra_curve[0], extra_curve[-1]):
            extra_curve = np.vstack([extra_curve, extra_curve[0]])

        ax.plot(
            extra_curve[:, 0],
            extra_curve[:, 1],
            '--',
            color='gray',
            linewidth=1.5,
            label="Spawn curve"
        )

    ax.plot(centroid[0], centroid[1], 'kx', markersize=8, label="Centroid")

    first_defender = True
    for defender in defenders:
        draw_drone(
            ax,
            defender["pos"],
            color="blue",
            size=drone_size
        )

        if first_defender:
            ax.plot([], [], 'bo', label="Defenders")
            first_defender = False

        if defender["target_id"] is not None:
            target = next((a for a in attackers if a["id"] == defender["target_id"]), None)

            if target is not None and target["active"]:
                ax.plot(
                    [defender["pos"][0], target["pos"][0]],
                    [defender["pos"][1], target["pos"][1]],
                    'b--',
                    linewidth=1.0,
                    alpha=0.45
                )

    first_active = True
    first_captured = True
    first_breached = True
    first_future = True
    first_sensing = True

    for attacker in attackers:
        if attacker["captured"]:
            ax.plot(
                attacker["pos"][0],
                attacker["pos"][1],
                'co',
                markersize=7,
                label="Captured" if first_captured else None
            )
            first_captured = False

        elif attacker["breached"]:
            ax.plot(
                attacker["closest_point"][0],
                attacker["closest_point"][1],
                'ko',
                markersize=6,
                label="Breached" if first_breached else None
            )
            first_breached = False

        elif attacker["active"]:
            draw_drone(
                ax,
                attacker["pos"],
                color="red",
                size=drone_size
            )

            if first_active:
                ax.plot([], [], 'ro', label="Active attackers")
                first_active = False

            sensing_circle = circle_from_center_radius(attacker["pos"], rhoA, n_plot=60)
            ax.plot(
                sensing_circle[:, 0],
                sensing_circle[:, 1],
                'r-',
                linewidth=1.0,
                alpha=0.35,
                label="Attacker sensing boundary" if first_sensing else None
            )
            first_sensing = False

        elif not attacker["spawned"]:
            ax.plot(
                attacker["start"][0],
                attacker["start"][1],
                '.',
                color='gray',
                markersize=5,
                label="Not yet arrived" if first_future else None
            )
            first_future = False

    ax.set_title(f"Time = {current_time:.2f}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    if fixed_xlim is not None:
        ax.set_xlim(fixed_xlim)

    if fixed_ylim is not None:
        ax.set_ylim(fixed_ylim)

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    clean_handles = []
    clean_labels = []

    for h, l in zip(handles, labels):
        if l and l not in seen:
            clean_handles.append(h)
            clean_labels.append(l)
            seen.add(l)

    ax.legend(clean_handles, clean_labels, loc="upper right")
    return fig


def plot_capture_percentage(time_history, capture_percentage_history):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(time_history, capture_percentage_history, linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Capture percentage")
    ax.set_title("Capture Percentage Over Time")
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True)

    return fig


# ============================================================
# Session defaults
# ============================================================
if "spawn_mode" not in st.session_state:
    st.session_state.spawn_mode = "Random around the boundary"

if "arrival_mode" not in st.session_state:
    st.session_state.arrival_mode = "Sequential"

if "defender_strategy" not in st.session_state:
    st.session_state.defender_strategy = "Optimal defender"


# ============================================================
# Streamlit app
# ============================================================
st.title("Multi-Defender Apollonius Target-Defense Game")

st.write(
    "Choose between the restored Apollonius-based optimal defender policy and the pure-pursuit defender policy."
)

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])


# ----------------------------
# Controls
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    num_boundary_samples = st.slider("Number of sampled boundary points", 50, 1200, 250, 10)
    threshold = st.slider("Threshold", 0, 255, 127, 1)
    invert = st.checkbox("Invert threshold", value=True)

with col2:
    nu = st.number_input("Attacker speed nu", min_value=0.01, max_value=0.99, value=0.50, step=0.01)
    defender_speed = st.number_input("Defender speed", min_value=0.01, max_value=5.0, value=1.0, step=0.05)
    rhoA = st.number_input("Attacker sensing radius rhoA", min_value=0.01, value=20.00, step=1.0)


col3, col4 = st.columns(2)

with col3:
    n_attackers = st.slider("Number of attackers", 1, 300, 30, 1)
    n_defenders = st.slider("Number of defenders", 1, 20, 2, 1)
    defender_spacing = st.number_input("Initial defender spacing", min_value=0.0, value=5.0, step=1.0)

with col4:
    n_future_samples = st.slider("Future trajectory samples", 3, 12, 4, 1)
    n_sensing_samples = st.slider("Sensing-boundary samples", 4, 24, 8, 1)
    n_circle_samples = st.slider("Apollonius circle samples", 30, 120, 60, 10)
    max_candidates = st.slider("Max attackers evaluated per free defender", 1, 30, 8, 1)


col5, col6 = st.columns(2)

with col5:
    total_time = st.number_input("Total simulation time", min_value=5.0, value=60.0, step=5.0)
    dt = st.number_input("Time step", min_value=0.01, value=0.20, step=0.01)

with col6:
    playback_delay = st.number_input("Display delay", min_value=0.0, value=0.00, step=0.01)
    random_seed = st.number_input("Random seed", min_value=0, value=0, step=1)
    drone_size = st.number_input("Drone icon size", min_value=0.5, value=4.0, step=0.5)



# ----------------------------
# Defender strategy buttons
# ----------------------------
st.subheader("Defender strategy")

ds1, ds2 = st.columns(2)

with ds1:
    if st.button("Optimal defender", use_container_width=True):
        st.session_state.defender_strategy = "Optimal defender"

with ds2:
    if st.button("Pure pursuit", use_container_width=True):
        st.session_state.defender_strategy = "Pure pursuit"

st.write(f"Current defender strategy: **{st.session_state.defender_strategy}**")

# ----------------------------
# Spawn geometry buttons
# ----------------------------
st.subheader("Spawn geometry")

b1, b2, b3 = st.columns(3)

with b1:
    if st.button("Random around boundary", use_container_width=True):
        st.session_state.spawn_mode = "Random around the boundary"

with b2:
    if st.button("Big circle perimeter", use_container_width=True):
        st.session_state.spawn_mode = "On perimeter of a big circle"

with b3:
    if st.button("Bigger same-shape boundary", use_container_width=True):
        st.session_state.spawn_mode = "On perimeter of a bigger same-shape boundary"

st.write(f"Current spawn mode: **{st.session_state.spawn_mode}**")

if st.session_state.spawn_mode == "Random around the boundary":
    spawn_size = st.number_input("Sampling area size", min_value=1.0, value=40.0, step=5.0)
elif st.session_state.spawn_mode == "On perimeter of a big circle":
    spawn_size = st.number_input("Circle radius increase", min_value=1.0, value=40.0, step=5.0)
else:
    spawn_size = st.number_input("Boundary scaling factor", min_value=1.01, value=1.25, step=0.05)


# ----------------------------
# Arrival pattern buttons
# ----------------------------
st.subheader("Arrival pattern")

a1, a2, a3 = st.columns(3)

with a1:
    if st.button("Sequential", use_container_width=True):
        st.session_state.arrival_mode = "Sequential"

with a2:
    if st.button("Periodic", use_container_width=True):
        st.session_state.arrival_mode = "Periodic"

with a3:
    if st.button("Random", use_container_width=True):
        st.session_state.arrival_mode = "Random"

st.write(f"Current arrival pattern: **{st.session_state.arrival_mode}**")

if st.session_state.arrival_mode == "Sequential":
    arrival_param = None
    st.caption("One attacker appears. The next appears after the current one is captured or breaches.")
elif st.session_state.arrival_mode == "Periodic":
    arrival_param = st.number_input("Period T", min_value=0.1, value=4.0, step=0.1)
else:
    arrival_param = st.number_input("Random arrival rate lambda", min_value=0.01, value=0.35, step=0.05)


play_button = st.button("Play Game")


# ============================================================
# Main app body
# ============================================================
if uploaded_file is not None:
    sampled_img, contour_img, image_rgb, binary, image_height = extract_edge_points_from_uploaded_image(
        uploaded_file=uploaded_file,
        num_samples=num_boundary_samples,
        threshold=threshold,
        invert=invert
    )

    if sampled_img is None or contour_img is None:
        st.error("No closed contour was found in the image.")
    else:
        boundary_cart = image_to_cartesian(sampled_img, image_height)
        boundary_data = make_boundary_data(boundary_cart)
        centroid = polygon_centroid(boundary_cart)

        st.subheader("Environment")
        st.image(image_rgb, use_container_width=True)

        if st.session_state.spawn_mode == "Random around the boundary":
            starts, extra_curve = generate_attackers_random_outside_box(
                boundary_points=boundary_cart,
                n_attackers=n_attackers,
                offset_margin=spawn_size,
                seed=int(random_seed)
            )

        elif st.session_state.spawn_mode == "On perimeter of a big circle":
            starts, extra_curve = generate_attackers_on_big_circle(
                boundary_points=boundary_cart,
                n_attackers=n_attackers,
                radius_offset=spawn_size,
                seed=int(random_seed)
            )

        else:
            starts, extra_curve = generate_attackers_on_scaled_boundary(
                boundary_points=boundary_cart,
                n_attackers=n_attackers,
                scale_factor=spawn_size,
                seed=int(random_seed)
            )

        all_pts = [boundary_cart, starts]
        if extra_curve is not None:
            all_pts.append(np.asarray(extra_curve))
        all_pts = np.vstack(all_pts)

        xmin, ymin = np.min(all_pts, axis=0)
        xmax, ymax = np.max(all_pts, axis=0)

        pad = 0.08 * max(xmax - xmin, ymax - ymin) + rhoA + drone_size + 1.0

        fixed_xlim = (xmin - pad, xmax + pad)
        fixed_ylim = (ymin - pad, ymax + pad)

        if play_button:
            attackers = initialize_attackers(boundary_data=boundary_data, starts=starts)

            defenders = initialize_defenders(
                centroid=centroid,
                n_defenders=n_defenders,
                spacing=defender_spacing
            )

            next_spawn_idx = 0
            rng = np.random.default_rng(int(random_seed))

            next_periodic_time = 0.0
            next_random_time = (
                rng.exponential(1.0 / float(arrival_param))
                if st.session_state.arrival_mode == "Random"
                else None
            )

            frame_slot = st.empty()
            info_slot = st.empty()
            progress_bar = st.progress(0.0)

            time_history = []
            capture_percentage_history = []

            t = 0.0
            n_steps = int(np.ceil(float(total_time) / float(dt)))

            for step in range(n_steps + 1):

                # ------------------------------------------------
                # Arrival logic
                # ------------------------------------------------
                if st.session_state.arrival_mode == "Sequential":
                    active_exists = any(a["active"] for a in attackers if not a["eliminated"])

                    if not active_exists and next_spawn_idx < len(attackers):
                        attackers[next_spawn_idx]["active"] = True
                        attackers[next_spawn_idx]["spawned"] = True
                        next_spawn_idx += 1

                elif st.session_state.arrival_mode == "Periodic":
                    while next_spawn_idx < len(attackers) and t >= next_periodic_time:
                        attackers[next_spawn_idx]["active"] = True
                        attackers[next_spawn_idx]["spawned"] = True
                        next_spawn_idx += 1
                        next_periodic_time += float(arrival_param)

                elif st.session_state.arrival_mode == "Random":
                    while next_spawn_idx < len(attackers) and t >= next_random_time:
                        attackers[next_spawn_idx]["active"] = True
                        attackers[next_spawn_idx]["spawned"] = True
                        next_spawn_idx += 1
                        next_random_time += rng.exponential(1.0 / float(arrival_param))

                # ------------------------------------------------
                # Move active attackers not in capture phase
                # ------------------------------------------------
                targets_in_capture = {
                    d["target_id"]
                    for d in defenders
                    if d["mode"] in ["move_to_capture", "pure_pursuit_capture"] and d["target_id"] is not None
                }

                for attacker in attackers:
                    if (
                        attacker["active"]
                        and not attacker["captured"]
                        and not attacker["breached"]
                        and not attacker["eliminated"]
                    ):
                        if attacker["id"] in targets_in_capture:
                            continue

                        attacker["pos"], arrived = move_toward(
                            attacker["pos"],
                            attacker["closest_point"],
                            nu,
                            float(dt)
                        )

                        if arrived:
                            attacker["breached"] = True
                            attacker["active"] = False
                            attacker["eliminated"] = True
                            attacker["assigned_defender_id"] = None
                            release_defender_from_attacker(defenders, attacker["id"])

                # ------------------------------------------------
                # Defender policy
                # ------------------------------------------------
                if st.session_state.defender_strategy == "Optimal defender":
                    # Restored optimal strategy from the uploaded version.
                    assign_free_defenders_apollonius_policy_fast(
                        defenders=defenders,
                        attackers=attackers,
                        boundary_data=boundary_data,
                        nu=nu,
                        rhoA=rhoA,
                        n_future_samples=n_future_samples,
                        n_sensing_samples=n_sensing_samples,
                        n_circle_samples=n_circle_samples,
                        defender_speed=defender_speed,
                        max_candidates=max_candidates
                    )

                    update_defenders_apollonius_game(
                        defenders=defenders,
                        attackers=attackers,
                        centroid=centroid,
                        defender_speed=defender_speed,
                        attacker_speed=nu,
                        dt=float(dt)
                    )
                else:
                    assign_pure_pursuit_defenders(
                        defenders=defenders,
                        attackers=attackers
                    )

                    update_defenders_pure_pursuit(
                        defenders=defenders,
                        attackers=attackers,
                        centroid=centroid,
                        boundary_data=boundary_data,
                        nu=nu,
                        rhoA=rhoA,
                        defender_speed=defender_speed,
                        dt=float(dt)
                    )

                # ------------------------------------------------
                # Counts and capture percentage
                # ------------------------------------------------
                n_active = sum(a["active"] for a in attackers)
                n_captured = sum(a["captured"] for a in attackers)
                n_breached = sum(a["breached"] for a in attackers)
                n_not_spawned = sum(not a["spawned"] for a in attackers)

                n_free_defenders = sum(d["target_id"] is None for d in defenders)
                n_busy_defenders = len(defenders) - n_free_defenders

                denom = n_captured + n_breached
                capture_percentage = n_captured / denom if denom > 0 else 0.0

                time_history.append(t)
                capture_percentage_history.append(capture_percentage)

                # ------------------------------------------------
                # Draw game
                # ------------------------------------------------
                fig = draw_game(
                    boundary_data=boundary_data,
                    defenders=defenders,
                    attackers=attackers,
                    centroid=centroid,
                    rhoA=rhoA,
                    drone_size=drone_size,
                    current_time=t,
                    extra_curve=extra_curve,
                    fixed_xlim=fixed_xlim,
                    fixed_ylim=fixed_ylim
                )

                frame_slot.pyplot(fig)
                plt.close(fig)

                info_slot.markdown(
                    f"""
**Defender strategy:** {st.session_state.defender_strategy}  
**Time:** {t:.2f}  
**Active attackers:** {n_active}  
**Captured attackers:** {n_captured}  
**Breached attackers:** {n_breached}  
**Capture percentage:** {100 * capture_percentage:.2f}%  
**Not yet arrived:** {n_not_spawned}  
**Free defenders:** {n_free_defenders}  
**Busy defenders:** {n_busy_defenders}
"""
                )

                progress_bar.progress(min((step + 1) / max(n_steps, 1), 1.0))

                if float(playback_delay) > 0:
                    time.sleep(float(playback_delay))

                t += float(dt)

            st.success("Game finished.")

            st.subheader("Capture Percentage Over Time")
            fig_metric = plot_capture_percentage(
                time_history=time_history,
                capture_percentage_history=capture_percentage_history
            )
            st.pyplot(fig_metric)
            plt.close(fig_metric)
