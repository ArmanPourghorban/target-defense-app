import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time


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

    sampled_points = np.array(sampled_points, dtype=float)

    return sampled_points, contour, image_rgb, binary, image_rgb.shape[0]


def image_to_cartesian(points, image_height):
    points = np.asarray(points, dtype=float).copy()
    points[:, 1] = image_height - points[:, 1]
    return points


# ============================================================
# Geometry utilities
# ============================================================
def point_to_segment_distance(point, a, b):
    point = np.asarray(point, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    ab = b - a
    ab_norm_sq = np.dot(ab, ab)

    if ab_norm_sq == 0:
        closest = a
        dist = np.linalg.norm(point - closest)
        return closest, dist

    t = np.dot(point - a, ab) / ab_norm_sq
    t = np.clip(t, 0.0, 1.0)

    closest = a + t * ab
    dist = np.linalg.norm(point - closest)

    return closest, dist


def point_to_polygon_distance(point, boundary_points):
    boundary_points = np.asarray(boundary_points, dtype=float)
    if not np.allclose(boundary_points[0], boundary_points[-1]):
        boundary_points = np.vstack([boundary_points, boundary_points[0]])

    dmin = np.inf
    for i in range(len(boundary_points) - 1):
        _, d = point_to_segment_distance(point, boundary_points[i], boundary_points[i + 1])
        dmin = min(dmin, d)
    return dmin


def closest_point_on_polygon(point, boundary_points):
    boundary_points = np.asarray(boundary_points, dtype=float)

    if not np.allclose(boundary_points[0], boundary_points[-1]):
        boundary_points = np.vstack([boundary_points, boundary_points[0]])

    min_dist = np.inf
    best_point = None
    best_segment_index = None

    for i in range(len(boundary_points) - 1):
        a = boundary_points[i]
        b = boundary_points[i + 1]
        candidate_point, candidate_dist = point_to_segment_distance(point, a, b)

        if candidate_dist < min_dist:
            min_dist = candidate_dist
            best_point = candidate_point
            best_segment_index = i

    return best_point, min_dist, best_segment_index


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

    cx /= (6 * a)
    cy /= (6 * a)
    return np.array([cx, cy])


# ============================================================
# Sensing / circles / Apollonius
# ============================================================
def sample_circle_points(center, radius, n_points=16):
    center = np.asarray(center, dtype=float)
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    pts = np.column_stack([
        center[0] + radius * np.cos(theta),
        center[1] + radius * np.sin(theta)
    ])
    return pts


def circle_from_center_radius(center, radius, n_plot=100):
    theta = np.linspace(0, 2 * np.pi, n_plot)
    circle = np.column_stack([
        center[0] + radius * np.cos(theta),
        center[1] + radius * np.sin(theta)
    ])
    return circle


def apollonius_circle(attacker_pos, sensing_point, nu):
    A = np.asarray(attacker_pos, dtype=float)
    S = np.asarray(sensing_point, dtype=float)

    denom = 1.0 - nu**2
    if abs(denom) < 1e-12:
        raise ValueError("nu is too close to 1. Choose nu < 1.")

    center = (A - (nu**2) * S) / denom
    radius = (nu * np.linalg.norm(A - S)) / denom
    return center, radius


def circle_intersects_polygon(center, radius, boundary_points, tol=1e-6):
    boundary_points = np.asarray(boundary_points, dtype=float)

    if not np.allclose(boundary_points[0], boundary_points[-1]):
        boundary_points = np.vstack([boundary_points, boundary_points[0]])

    for i in range(len(boundary_points) - 1):
        a = boundary_points[i]
        b = boundary_points[i + 1]
        _, dist = point_to_segment_distance(center, a, b)
        if dist <= radius + tol:
            return True

    return False


def apollonius_circle_worst_distance(center, radius, boundary_points, n_circle_samples=90):
    circle_pts = sample_circle_points(center, radius, n_points=n_circle_samples)
    dists = np.array([point_to_polygon_distance(p, boundary_points) for p in circle_pts])
    max_idx = int(np.argmax(dists))
    worst_point = circle_pts[max_idx]
    max_dist = float(dists[max_idx])
    return max_dist, worst_point


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
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        p = np.array([x, y], dtype=float)

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
    max_trials = 50000
    trials = 0

    while len(attackers) < n_attackers and trials < max_trials:
        theta = rng.uniform(0, 2 * np.pi)
        p = centroid + R * np.array([np.cos(theta), np.sin(theta)])

        if not is_point_inside_polygon(p, boundary_points):
            attackers.append(p)

        trials += 1

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
# Arrival patterns
# ============================================================
def initialize_attackers(boundary_points, starts):
    attackers = []
    for i, start in enumerate(starts):
        closest_pt, _, _ = closest_point_on_polygon(start, boundary_points)
        attackers.append({
            "id": i,
            "start": np.array(start, dtype=float),
            "pos": np.array(start, dtype=float),
            "closest_point": np.array(closest_pt, dtype=float),
            "active": False,
            "captured": False,
            "breached": False,
            "eliminated": False,
            "spawned": False
        })
    return attackers


# ============================================================
# Hidden-approach constraint
# ============================================================
def defender_hidden_until_engagement(
    defender_start,
    engagement_point,
    attacker_start_now,
    attacker_sample_point,
    rhoA,
    defender_speed,
    attacker_speed,
    engagement_time_from_now,
    n_samples=18
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
# Best attacker evaluation (locked once chosen)
# ============================================================
def evaluate_attacker(
    attacker,
    defender_pos,
    boundary_points,
    nu,
    rhoA,
    n_future_samples=5,
    n_sensing_samples=10,
    n_circle_samples=90,
    defender_speed=1.0
):
    if attacker["captured"] or attacker["breached"] or attacker["eliminated"] or (not attacker["active"]):
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
        A_sample = point_along_segment_by_time(A_now, target_pt, nu, tau)
        sensing_pts = sample_circle_points(A_sample, rhoA, n_points=n_sensing_samples)

        for S in sensing_pts:
            defender_time = np.linalg.norm(defender_pos - S) / defender_speed
            attacker_time = tau

            if defender_time < attacker_time:
                hidden_ok = defender_hidden_until_engagement(
                    defender_start=defender_pos,
                    engagement_point=S,
                    attacker_start_now=A_now,
                    attacker_sample_point=A_sample,
                    rhoA=rhoA,
                    defender_speed=defender_speed,
                    attacker_speed=nu,
                    engagement_time_from_now=tau,
                    n_samples=18
                )

                if not hidden_ok:
                    continue

                center, radius = apollonius_circle(A_sample, S, nu)
                intersects = circle_intersects_polygon(center, radius, boundary_points)

                if not intersects:
                    score, worst_point = apollonius_circle_worst_distance(
                        center, radius, boundary_points, n_circle_samples=n_circle_samples
                    )

                    capture_leg = np.linalg.norm(np.asarray(S) - np.asarray(worst_point)) / defender_speed
                    total_capture_time = attacker_time + capture_leg

                    if score < best_score:
                        best_score = score
                        best_data = {
                            "attacker_id": attacker["id"],
                            "attacker_sample_point": A_sample.copy(),
                            "engagement_point": np.array(S, dtype=float),
                            "capture_point": np.array(worst_point, dtype=float),
                            "score": float(score),
                            "attacker_sample_time_rel": float(attacker_time),
                            "capture_finish_time_rel": float(total_capture_time)
                        }

    return best_data


def choose_best_attacker(
    attackers,
    defender_pos,
    boundary_points,
    nu,
    rhoA,
    n_future_samples=5,
    n_sensing_samples=10,
    n_circle_samples=90,
    defender_speed=1.0
):
    results = []
    for attacker in attackers:
        res = evaluate_attacker(
            attacker=attacker,
            defender_pos=defender_pos,
            boundary_points=boundary_points,
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


# ============================================================
# Plotting
# ============================================================
def draw_game(
    boundary_points,
    defender_pos,
    attackers,
    centroid,
    rhoA,
    current_time=0.0,
    extra_curve=None,
    fixed_xlim=None,
    fixed_ylim=None
):
    fig, ax = plt.subplots(figsize=(8, 8))

    boundary_points = np.asarray(boundary_points, dtype=float)
    if not np.allclose(boundary_points[0], boundary_points[-1]):
        boundary_closed = np.vstack([boundary_points, boundary_points[0]])
    else:
        boundary_closed = boundary_points.copy()

    ax.plot(boundary_closed[:, 0], boundary_closed[:, 1], 'k-', linewidth=2.5, label='Boundary')

    if extra_curve is not None:
        extra_curve = np.asarray(extra_curve, dtype=float)
        if not np.allclose(extra_curve[0], extra_curve[-1]):
            extra_curve = np.vstack([extra_curve, extra_curve[0]])
        ax.plot(extra_curve[:, 0], extra_curve[:, 1], '--', color='gray', linewidth=1.5, label='Spawn curve')

    ax.plot(centroid[0], centroid[1], 'kx', markersize=8, label='Centroid')
    ax.plot(defender_pos[0], defender_pos[1], 'bo', markersize=10, label='Defender')

    first_active = True
    first_captured = True
    first_breached = True
    first_future = True
    first_sensing = True

    for attacker in attackers:
        if attacker["captured"]:
            if first_captured:
                ax.plot(attacker["pos"][0], attacker["pos"][1], 'co', markersize=7, label='Captured')
                first_captured = False
            else:
                ax.plot(attacker["pos"][0], attacker["pos"][1], 'co', markersize=7)

        elif attacker["breached"]:
            if first_breached:
                ax.plot(attacker["closest_point"][0], attacker["closest_point"][1], 'ko', markersize=6, label='Breached')
                first_breached = False
            else:
                ax.plot(attacker["closest_point"][0], attacker["closest_point"][1], 'ko', markersize=6)

        elif attacker["active"]:
            if first_active:
                ax.plot(attacker["pos"][0], attacker["pos"][1], 'ro', markersize=7, label='Active attackers')
                first_active = False
            else:
                ax.plot(attacker["pos"][0], attacker["pos"][1], 'ro', markersize=7)

            sensing_circle = circle_from_center_radius(attacker["pos"], rhoA, n_plot=80)
            if first_sensing:
                ax.plot(sensing_circle[:, 0], sensing_circle[:, 1], 'r-', linewidth=1.2, alpha=0.7, label='Attacker sensing boundary')
                first_sensing = False
            else:
                ax.plot(sensing_circle[:, 0], sensing_circle[:, 1], 'r-', linewidth=1.0, alpha=0.45)

        elif not attacker["spawned"]:
            if first_future:
                ax.plot(attacker["start"][0], attacker["start"][1], '.', color='gray', markersize=5, label='Not yet arrived')
                first_future = False
            else:
                ax.plot(attacker["start"][0], attacker["start"][1], '.', color='gray', markersize=5)

    ax.set_title(f"Time = {current_time:.2f}")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    if fixed_xlim is not None:
        ax.set_xlim(fixed_xlim)
    if fixed_ylim is not None:
        ax.set_ylim(fixed_ylim)

    ax.legend(loc='upper right')

    return fig


# ============================================================
# Session-state defaults
# ============================================================
if "spawn_mode" not in st.session_state:
    st.session_state.spawn_mode = "Random around the boundary"

if "arrival_mode" not in st.session_state:
    st.session_state.arrival_mode = "Sequential"


# ============================================================
# Streamlit app
# ============================================================
st.title("Target-Defense Game")

st.write("Upload an environment, choose spawn geometry and arrival pattern, then press Play Game.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])

col1, col2 = st.columns(2)
with col1:
    num_boundary_samples = st.slider("Number of sampled boundary points", 50, 1200, 250, 10)
    threshold = st.slider("Threshold", 0, 255, 127, 1)
    invert = st.checkbox("Invert threshold", value=True)
with col2:
    nu = st.number_input("Attacker speed nu", min_value=0.01, max_value=0.99, value=0.80, step=0.01)
    rhoA = st.number_input("Attacker sensing radius rhoA", min_value=0.01, value=0.70, step=0.05)
    n_attackers = st.slider("Number of attackers", 1, 200, 30, 1)

col3, col4 = st.columns(2)
with col3:
    n_future_samples = st.slider("Future trajectory samples", 3, 15, 5, 1)
    n_sensing_samples = st.slider("Sensing-boundary samples", 4, 32, 10, 1)
with col4:
    n_circle_samples = st.slider("Apollonius circle samples", 40, 160, 90, 10)
    random_seed = st.number_input("Random seed", min_value=0, value=0, step=1)

st.subheader("Spawn geometry")
b1, b2, b3 = st.columns(3)
with b1:
    if st.button("Random around the boundary", use_container_width=True):
        st.session_state.spawn_mode = "Random around the boundary"
with b2:
    if st.button("On a big circle", use_container_width=True):
        st.session_state.spawn_mode = "On perimeter of a big circle"
with b3:
    if st.button("On bigger same-shape boundary", use_container_width=True):
        st.session_state.spawn_mode = "On perimeter of a bigger same-shape boundary"

st.write(f"Current spawn mode: **{st.session_state.spawn_mode}**")

if st.session_state.spawn_mode == "Random around the boundary":
    spawn_size = st.number_input("Sampling area size (expanded box margin)", min_value=1.0, value=40.0, step=5.0)
elif st.session_state.spawn_mode == "On perimeter of a big circle":
    spawn_size = st.number_input("Circle radius increase", min_value=1.0, value=40.0, step=5.0)
else:
    spawn_size = st.number_input("Boundary scaling factor", min_value=1.01, value=1.25, step=0.05)

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
    st.caption("One attacker at a time. The next one appears only after the current one is captured or breaches.")
elif st.session_state.arrival_mode == "Periodic":
    arrival_param = st.number_input("Period T", min_value=0.1, value=4.0, step=0.1)
else:
    arrival_param = st.number_input("Random arrival rate λ", min_value=0.01, value=0.35, step=0.05)

col5, col6 = st.columns(2)
with col5:
    total_time = st.number_input("Total simulation time", min_value=5.0, value=60.0, step=5.0)
    dt = st.number_input("Time step", min_value=0.01, value=0.20, step=0.01)
with col6:
    playback_delay = st.number_input("Display delay (seconds)", min_value=0.0, value=0.01, step=0.01)
    defender_speed = st.number_input("Defender speed", min_value=0.01, value=1.0, step=0.05)

play_button = st.button("Play Game")

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
        centroid = polygon_centroid(boundary_cart)

        st.subheader("Environment")
        st.image(image_rgb, use_container_width=True)

        default_def_x = float(centroid[0])
        default_def_y = float(centroid[1])

        d1, d2 = st.columns(2)
        with d1:
            defender_x = st.number_input("Defender initial x", value=default_def_x)
        with d2:
            defender_y = st.number_input("Defender initial y", value=default_def_y)

        defender_start = np.array([defender_x, defender_y], dtype=float)

        # Build attacker starts
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

        # Fixed axes so nothing shifts during simulation
        all_pts = [boundary_cart, starts, np.array([defender_start])]
        if extra_curve is not None:
            all_pts.append(np.asarray(extra_curve))
        all_pts = np.vstack(all_pts)

        xmin, ymin = np.min(all_pts, axis=0)
        xmax, ymax = np.max(all_pts, axis=0)
        pad = 0.08 * max(xmax - xmin, ymax - ymin) + rhoA + 1.0
        fixed_xlim = (xmin - pad, xmax + pad)
        fixed_ylim = (ymin - pad, ymax + pad)

        if play_button:
            attackers = initialize_attackers(boundary_points=boundary_cart, starts=starts)

            next_spawn_idx = 0
            rng = np.random.default_rng(int(random_seed))
            next_periodic_time = 0.0
            next_random_time = rng.exponential(1.0 / arrival_param) if st.session_state.arrival_mode == "Random" else None

            defender_pos = defender_start.copy()

            frame_slot = st.empty()
            info_slot = st.empty()
            progress_bar = st.progress(0.0)

            mode = "seek"
            current_plan = None
            current_target_id = None

            t = 0.0
            n_steps = int(np.ceil(float(total_time) / float(dt)))

            for step in range(n_steps + 1):
                # Arrival logic
                if st.session_state.arrival_mode == "Sequential":
                    active_exists = any(a["active"] for a in attackers if not a["eliminated"])
                    if (not active_exists) and (next_spawn_idx < len(attackers)):
                        attackers[next_spawn_idx]["active"] = True
                        attackers[next_spawn_idx]["spawned"] = True
                        next_spawn_idx += 1

                elif st.session_state.arrival_mode == "Periodic":
                    while (next_spawn_idx < len(attackers)) and (t >= next_periodic_time):
                        attackers[next_spawn_idx]["active"] = True
                        attackers[next_spawn_idx]["spawned"] = True
                        next_spawn_idx += 1
                        next_periodic_time += float(arrival_param)

                elif st.session_state.arrival_mode == "Random":
                    while (next_spawn_idx < len(attackers)) and (t >= next_random_time):
                        attackers[next_spawn_idx]["active"] = True
                        attackers[next_spawn_idx]["spawned"] = True
                        next_spawn_idx += 1
                        next_random_time += rng.exponential(1.0 / float(arrival_param))

                # Move non-target attackers
                for attacker in attackers:
                    if attacker["active"] and (not attacker["captured"]) and (not attacker["breached"]) and (not attacker["eliminated"]):
                        if current_target_id is not None and attacker["id"] == current_target_id and mode == "move_to_capture":
                            continue

                        attacker["pos"], arrived = move_toward(attacker["pos"], attacker["closest_point"], nu, float(dt))
                        if arrived:
                            attacker["breached"] = True
                            attacker["active"] = False
                            attacker["eliminated"] = True

                # Reset if target vanished
                if current_target_id is not None:
                    target = next((a for a in attackers if a["id"] == current_target_id), None)
                    if target is None or target["captured"] or target["breached"] or target["eliminated"]:
                        current_target_id = None
                        current_plan = None
                        mode = "seek"

                # LOCKED TARGET: choose best only when no target is assigned
                if current_target_id is None and mode in {"seek", "move_to_centroid"}:
                    best = choose_best_attacker(
                        attackers=attackers,
                        defender_pos=defender_pos,
                        boundary_points=boundary_cart,
                        nu=nu,
                        rhoA=rhoA,
                        n_future_samples=n_future_samples,
                        n_sensing_samples=n_sensing_samples,
                        n_circle_samples=n_circle_samples,
                        defender_speed=defender_speed
                    )

                    if best is not None:
                        current_plan = best
                        current_target_id = best["attacker_id"]
                        mode = "move_to_engagement"
                    else:
                        mode = "move_to_centroid"

                # Defender state machine
                if mode == "move_to_engagement" and current_plan is not None:
                    defender_pos, arrived = move_toward(defender_pos, current_plan["engagement_point"], defender_speed, float(dt))
                    current_plan["attacker_sample_time_rel"] = max(0.0, current_plan["attacker_sample_time_rel"] - float(dt))
                    current_plan["capture_finish_time_rel"] = max(0.0, current_plan["capture_finish_time_rel"] - float(dt))
                    if arrived:
                        mode = "wait_for_attacker"

                elif mode == "wait_for_attacker" and current_plan is not None:
                    current_plan["attacker_sample_time_rel"] = max(0.0, current_plan["attacker_sample_time_rel"] - float(dt))
                    current_plan["capture_finish_time_rel"] = max(0.0, current_plan["capture_finish_time_rel"] - float(dt))
                    if current_plan["attacker_sample_time_rel"] <= 0:
                        mode = "move_to_capture"

                elif mode == "move_to_capture" and current_plan is not None:
                    defender_pos, arrived_def = move_toward(defender_pos, current_plan["capture_point"], defender_speed, float(dt))

                    target = next((a for a in attackers if a["id"] == current_target_id), None)
                    if target is not None and target["active"] and (not target["captured"]) and (not target["breached"]):
                        target["pos"], arrived_att = move_toward(target["pos"], current_plan["capture_point"], nu, float(dt))
                    else:
                        arrived_att = False

                    if target is not None:
                        if np.linalg.norm(defender_pos - target["pos"]) < 1e-2 or arrived_def or arrived_att:
                            target["captured"] = True
                            target["active"] = False
                            target["eliminated"] = True
                            target["pos"] = current_plan["capture_point"].copy()

                            current_target_id = None
                            current_plan = None
                            mode = "seek"

                elif mode == "move_to_centroid":
                    defender_pos, arrived = move_toward(defender_pos, centroid, defender_speed, float(dt))
                    if arrived:
                        mode = "seek"

                # Draw frame
                fig = draw_game(
                    boundary_points=boundary_cart,
                    defender_pos=defender_pos,
                    attackers=attackers,
                    centroid=centroid,
                    rhoA=rhoA,
                    current_time=t,
                    extra_curve=extra_curve,
                    fixed_xlim=fixed_xlim,
                    fixed_ylim=fixed_ylim
                )
                frame_slot.pyplot(fig)
                plt.close(fig)

                n_active = sum(a["active"] for a in attackers)
                n_captured = sum(a["captured"] for a in attackers)
                n_breached = sum(a["breached"] for a in attackers)
                n_not_spawned = sum((not a["spawned"]) for a in attackers)

                info_slot.markdown(
                    f"""
**Mode:** {mode}  
**Time:** {t:.2f}  
**Active attackers:** {n_active}  
**Captured attackers:** {n_captured}  
**Breached attackers:** {n_breached}  
**Not yet arrived:** {n_not_spawned}
"""
                )

                progress_bar.progress(min((step + 1) / max(n_steps, 1), 1.0))

                if float(playback_delay) > 0:
                    time.sleep(float(playback_delay))
                t += float(dt)

            st.success("Game finished.")