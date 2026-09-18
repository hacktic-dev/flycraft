
import math
import os
import time
import random
import pygame
import flycraft_pictograms as pictograms

if "--self-test" in os.sys.argv or "--preview" in os.sys.argv:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

pygame.init()

BASE_W, BASE_H = 1600, 900
BG = (11, 17, 27)  # #0B111B
FG = (225, 230, 235)
DIM = (95, 105, 115)
BORDER = (55, 65, 75)
ACCENT = (115, 175, 255)
GOOD = (90, 215, 140)
BAD = (230, 100, 100)
TREE = (100, 180, 110)
FLY = (230, 220, 120)

INITIAL_HEADING = -65.0
GOOD_DURATION = 3.0
BAD_DURATION = 3.0

BASE_WEIGHTS = [0.72, 0.58, 0.81, 0.66, 0.74, 0.63]
ACTIVE_EDGES = {1, 2, 4}

screen = pygame.display.set_mode((BASE_W, BASE_H), pygame.RESIZABLE)
pygame.display.set_caption("FlyCraft Learning Visualizer")
clock = pygame.time.Clock()

FONT = pygame.font.SysFont("consolas", 28)
SMALL = pygame.font.SysFont("consolas", 21)

# Logical layout is always authored at 1600x900 and scaled to the actual window.
ARENA = pygame.Rect(28, 28, 790, 844)
BRAIN = pygame.Rect(842, 28, 730, 844)

fly_pos = pygame.Vector2(ARENA.centerx, ARENA.centery + 150)
tree_pos = pygame.Vector2(ARENA.centerx + 225, ARENA.centery - 210)

rng = random.Random(42)

# ----------------------------
# Utility
# ----------------------------

def clamp01(v):
    return max(0.0, min(1.0, v))

def smoothstep(t):
    t = clamp01(t)
    return t * t * (3.0 - 2.0 * t)

def lerp(a, b, t):
    return a + (b - a) * t

def wrap_angle(deg):
    while deg > 180:
        deg -= 360
    while deg < -180:
        deg += 360
    return deg

def angle_to_target(pos=None):
    p = fly_pos if pos is None else pos
    d = tree_pos - p
    return math.degrees(math.atan2(-d.y, d.x))

def heading_vector(deg, length):
    r = math.radians(deg)
    return pygame.Vector2(math.cos(r) * length, -math.sin(r) * length)

def text(surface, s, x, y, colour=FG, font=FONT):
    surface.blit(font.render(str(s), True, colour), (x, y))

# ----------------------------
# Fixed stage endpoints
# ----------------------------

TARGET_HEADING = angle_to_target()
GOOD_END_HEADING = lerp(INITIAL_HEADING, TARGET_HEADING, 0.72)

# Force stage 2 to turn away from the tree from the exact stage-1 endpoint.
delta_to_tree = wrap_angle(TARGET_HEADING - GOOD_END_HEADING)
BAD_DIRECTION = -1 if delta_to_tree > 0 else 1
BAD_END_HEADING = wrap_angle(GOOD_END_HEADING + BAD_DIRECTION * 50.0)

GOOD_END_WEIGHTS = BASE_WEIGHTS[:]
GOOD_END_WEIGHTS[1] += 0.13
GOOD_END_WEIGHTS[2] += 0.11
GOOD_END_WEIGHTS[4] += 0.15

BAD_END_WEIGHTS = GOOD_END_WEIGHTS[:]
BAD_END_WEIGHTS[1] -= 0.13
BAD_END_WEIGHTS[2] -= 0.11
BAD_END_WEIGHTS[4] -= 0.15

# ----------------------------
# App state
# ----------------------------

mode = "idle"
stage_time = 0.0
paused = False

current_heading = INITIAL_HEADING
weights = BASE_WEIGHTS[:]
signal_value = 0.0
signal_colour = DIM

floating_numbers = []
pulses = []
last_emit_time = -0.15

fullscreen = False
windowed_size = (BASE_W, BASE_H)

# Stage-7 state
learning_attempt = 0
learning_attempt_time = 0.0

# ----------------------------
# Drawing
# ----------------------------

def draw_tree(surface, pos):
    # Preserve the original target and overall footprint; share the exact new art.
    pictograms.tree(surface, pos.x, pos.y + 134, scale=0.86)


def draw_arrow(surface, origin, deg, length, colour, width=7):
    end = origin + heading_vector(deg, length)
    v = heading_vector(deg, 1)
    # Stop the shaft at the head's base so its thick cap cannot cross the tip.
    pygame.draw.line(surface, colour, origin, end - v * 22, width)
    left = end - v * 22 + pygame.Vector2(-v.y, v.x) * 13
    right = end - v * 22 - pygame.Vector2(-v.y, v.x) * 13
    pygame.draw.polygon(surface, colour, [end, left, right])

def draw_fly(surface, pos, heading=0.0, show_rotation=True, radius=52):
    pictograms.fly(surface, pos, r=radius, t=stage_time)
    if show_rotation:
        draw_arrow(surface, pos, heading, 160, ACCENT, 9)


def brain_geometry():
    # Attach the six existing trainable connections to actual cloud neurons.
    center = pygame.Vector2(BRAIN.center)
    points = [center + pygame.Vector2(p.x * 310, p.y * 310 * 0.73)
              for p in pictograms.NODES]
    targets = [(-0.65, -0.50), (-0.85, 0), (-0.65, 0.50),
               (0.65, -0.50), (0.85, 0), (0.65, 0.50)]
    nodes = []
    for x, y in targets:
        target = center + pygame.Vector2(x * 310, y * 310 * 0.73)
        nodes.append(min(points, key=lambda p: p.distance_squared_to(target)))
    edges = [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (2, 2)]
    return nodes[:3], nodes[3:], edges


def active_width(weight):
    # Purely continuous mapping. No sudden threshold jumps.
    return 4.0 + weight * 23.0

def inactive_width(weight):
    return 2.0 + weight * 5.0

def draw_brain(surface):
    pygame.draw.rect(surface, BORDER, BRAIN, 2)
    cloud = pygame.Surface((BASE_W, BASE_H), pygame.SRCALPHA)
    pictograms.neural(cloud, BRAIN.center, 310, stage_time)
    cloud.set_alpha(165)
    surface.blit(cloud, (0, 0))
    left_nodes, right_nodes, edges = brain_geometry()

    for i, (li, ri) in enumerate(edges):
        a = left_nodes[li]
        b = right_nodes[ri]
        w = weights[i]
        active = i in ACTIVE_EDGES
        colour = signal_colour if active and signal_colour != DIM else (ACCENT if active else BORDER)

        width_f = active_width(w) if active else inactive_width(w)
        pygame.draw.line(surface, colour, a, b, max(1, round(width_f * 0.35)))

    for p in left_nodes + right_nodes:
        pygame.draw.circle(surface, ACCENT, (round(p.x), round(p.y)), 7)
        pygame.draw.circle(surface, FG, (round(p.x), round(p.y)), 8, 2)

    text(surface, f"{signal_value:+.2f}", BRAIN.centerx - 38, BRAIN.bottom - 64,
         signal_colour, FONT)

def draw_brain_labels(surface):
    left_nodes, right_nodes, edges = brain_geometry()
    # Draw labels after ALL wires, away from the central crossings.
    for i, (li, ri) in enumerate(edges):
        a, b = left_nodes[li], right_nodes[ri]
        mid = a.lerp(b, 0.52 if i == 3 else (0.27 if i % 2 == 0 else 0.73))
        colour = signal_colour if i in ACTIVE_EDGES and signal_colour != DIM else (ACCENT if i in ACTIVE_EDGES else DIM)
        label = SMALL.render(f"{weights[i]:.3f}", True, colour)
        rect = label.get_rect(center=(round(mid.x), round(mid.y + (28 if i == 3 else (30 if i == 5 else -26)))))
        pygame.draw.rect(surface, BG, rect.inflate(12, 8), border_radius=4)
        surface.blit(label, rect)


def draw_arena(surface, show_main_fly=True, show_rotation=True):
    pygame.draw.rect(surface, BORDER, ARENA, 2)
    draw_tree(surface, tree_pos)

    if show_main_fly:
        draw_fly(surface, fly_pos, current_heading, show_rotation=show_rotation, radius=52)
        err = abs(wrap_angle(TARGET_HEADING - current_heading))
        text(surface, f"{current_heading:6.1f}°", ARENA.x + 32, ARENA.bottom - 82, DIM, SMALL)
        text(surface, f"{err:6.1f}°", ARENA.x + 32, ARENA.bottom - 46, FG, SMALL)

# ----------------------------
# Feedback particles
# ----------------------------

def reset_particles():
    global floating_numbers, pulses, last_emit_time
    floating_numbers = []
    pulses = []
    last_emit_time = -0.15

def emit_feedback_event(kind, magnitude, source_pos, born=0.0):
    colour = GOOD if kind == "reward" else BAD
    sign = "+" if magnitude >= 0 else ""

    floating_numbers.append({
        "text": f"{sign}{magnitude:.2f}",
        "pos": source_pos + pygame.Vector2(
            -96 + (round(born / 0.15) % 3) * 80,
            -96
        ),
        "born": born,
        "age": 0.0,
        "life": 0.95,
        "colour": colour,
    })

    left_nodes, right_nodes, edges = brain_geometry()
    edge_index = rng.choice(sorted(ACTIVE_EDGES))
    li, ri = edges[edge_index]
    target = left_nodes[li].lerp(right_nodes[ri], 0.56)

    pulses.append({
        "start": pygame.Vector2(BRAIN.x + 18, BRAIN.centery + rng.uniform(-130, 130)),
        "end": target,
        "born": born,
        "age": 0.0,
        "life": 0.72,
        "colour": colour,
    })

def emit_continuous(kind, t, source_pos, interval=0.15):
    global last_emit_time, signal_value, signal_colour
    signal_colour = GOOD if kind == "reward" else BAD

    # Deterministic catch-up loop. This keeps the cadence stable at low FPS.
    while t - last_emit_time >= interval - 1e-9:
        last_emit_time += interval
        if kind == "reward":
            magnitude = 0.08 + rng.uniform(0.0, 0.06)
        else:
            magnitude = -0.07 - rng.uniform(0.0, 0.05)
        signal_value = magnitude
        event_time = max(0.0, last_emit_time)
        event_pos = learning_pose(event_time)[0] if mode == "learning" else source_pos
        emit_feedback_event(kind, magnitude, event_pos, event_time)

def update_and_draw_particles(surface, dt):
    global floating_numbers, pulses

    next_numbers = []
    for f in floating_numbers:
        f["age"] = max(0.0, stage_time - f["born"])
        if f["age"] < f["life"]:
            frac = f["age"] / f["life"]
            pos = f["pos"] + pygame.Vector2(0, -100 * frac)
            alpha = int(255 * (1.0 - frac))
            surf = SMALL.render(f["text"], True, f["colour"])
            surf.set_alpha(alpha)
            surface.set_clip(ARENA.inflate(-8, -8))
            surface.blit(surf, pos)
            surface.set_clip(None)
            next_numbers.append(f)
    floating_numbers = next_numbers

    next_pulses = []
    for p in pulses:
        p["age"] = max(0.0, stage_time - p["born"])
        if p["age"] < p["life"]:
            frac = smoothstep(p["age"] / p["life"])
            pos = p["start"].lerp(p["end"], frac)
            pygame.draw.circle(surface, p["colour"], (int(pos.x), int(pos.y)), 10)
            next_pulses.append(p)
    pulses = next_pulses

# ----------------------------
# Deterministic stage setup
# ----------------------------

def start_stage(stage):
    global mode, stage_time, paused
    global current_heading, weights, signal_value, signal_colour
    global learning_attempt, learning_attempt_time

    mode = stage
    stage_time = 0.0
    paused = False
    signal_value = 0.0
    signal_colour = DIM
    learning_attempt = 0
    learning_attempt_time = 0.0
    reset_particles()
    rng.seed(42)

    if stage == "good":
        # Stage 1 ALWAYS resets to the true original state.
        current_heading = INITIAL_HEADING
        weights = BASE_WEIGHTS[:]

    elif stage == "bad":
        # Stage 2 ALWAYS begins at stage 1's exact endpoint.
        current_heading = GOOD_END_HEADING
        weights = GOOD_END_WEIGHTS[:]

    elif stage == "learning":
        # Stage 7 is independent and guaranteed to start.
        current_heading = INITIAL_HEADING
        weights = BASE_WEIGHTS[:]

    elif stage == "reward":
        current_heading = GOOD_END_HEADING
        weights = GOOD_END_WEIGHTS[:]

    elif stage == "punish":
        current_heading = BAD_END_HEADING
        weights = BAD_END_WEIGHTS[:]

    elif stage == "strengthen":
        current_heading = GOOD_END_HEADING
        weights = BASE_WEIGHTS[:]

    elif stage == "weaken":
        current_heading = BAD_END_HEADING
        weights = GOOD_END_WEIGHTS[:]

    else:
        current_heading = INITIAL_HEADING
        weights = BASE_WEIGHTS[:]

# ----------------------------
# Stage animation logic
# ----------------------------

def interpolate_weights(start, end, progress):
    p = smoothstep(progress)
    return [lerp(a, b, p) for a, b in zip(start, end)]

def update_stage_good(t):
    global current_heading, weights, signal_colour
    p = clamp01(t / GOOD_DURATION)

    current_heading = lerp(INITIAL_HEADING, GOOD_END_HEADING, smoothstep(p))
    weights = interpolate_weights(BASE_WEIGHTS, GOOD_END_WEIGHTS, p)
    signal_colour = GOOD

    emit_continuous("reward", min(t, GOOD_DURATION), fly_pos)

    # Once complete, values remain frozen at the endpoint.
    if p >= 1.0:
        current_heading = GOOD_END_HEADING
        weights = GOOD_END_WEIGHTS[:]

def update_stage_bad(t):
    global current_heading, weights, signal_colour
    p = clamp01(t / BAD_DURATION)

    # Fixed start/end values. No recomputation from changing state -> no flicker.
    current_heading = lerp(GOOD_END_HEADING, BAD_END_HEADING, smoothstep(p))
    weights = interpolate_weights(GOOD_END_WEIGHTS, BAD_END_WEIGHTS, p)
    signal_colour = BAD

    emit_continuous("punish", min(t, BAD_DURATION), fly_pos)

    if p >= 1.0:
        current_heading = BAD_END_HEADING
        weights = BAD_END_WEIGHTS[:]

def update_stage_reward(t):
    global signal_colour
    signal_colour = GOOD
    emit_continuous("reward", t, fly_pos)

def update_stage_punish(t):
    global signal_colour
    signal_colour = BAD
    emit_continuous("punish", t, fly_pos)

def update_stage_strengthen(t):
    global weights, signal_colour
    p = clamp01(t / 2.5)
    weights = interpolate_weights(BASE_WEIGHTS, GOOD_END_WEIGHTS, p)
    signal_colour = GOOD
    emit_continuous("reward", min(t, 2.5), fly_pos)

def update_stage_weaken(t):
    global weights, signal_colour
    p = clamp01(t / 2.5)
    weights = interpolate_weights(GOOD_END_WEIGHTS, BAD_END_WEIGHTS, p)
    signal_colour = BAD
    emit_continuous("punish", min(t, 2.5), fly_pos)

def bezier(points, t):
    work = [p.copy() for p in points]
    while len(work) > 1:
        work = [work[i].lerp(work[i+1], t) for i in range(len(work)-1)]
    return work[0]

def learning_path(attempt, local_t):
    start = pygame.Vector2(ARENA.x + 165, ARENA.bottom - 145)
    target = pygame.Vector2(tree_pos.x, tree_pos.y + 18)

    curves = [
        [
            start,
            pygame.Vector2(ARENA.x + 300, ARENA.bottom - 215),
            pygame.Vector2(ARENA.x + 230, ARENA.y + 300),
            pygame.Vector2(ARENA.x + 440, ARENA.y + 270),
        ],
        [
            start,
            pygame.Vector2(ARENA.x + 340, ARENA.bottom - 245),
            pygame.Vector2(ARENA.x + 420, ARENA.y + 345),
            pygame.Vector2(ARENA.x + 555, ARENA.y + 250),
        ],
        [
            start,
            pygame.Vector2(ARENA.x + 390, ARENA.bottom - 275),
            pygame.Vector2(ARENA.x + 520, ARENA.y + 330),
            pygame.Vector2(ARENA.x + 610, ARENA.y + 205),
        ],
        [
            start,
            pygame.Vector2(ARENA.x + 420, ARENA.bottom - 300),
            pygame.Vector2(ARENA.x + 565, ARENA.y + 300),
            target,
        ],
    ]

    return curves[attempt], target

ATTEMPT_DURATION = 2.4
FLIGHT_DURATION = 1.65
LEARNING_DURATION = ATTEMPT_DURATION * 3 + FLIGHT_DURATION


def learning_pose(t):
    t = max(0.0, min(t, LEARNING_DURATION))
    attempt = min(3, int(t / ATTEMPT_DURATION))
    elapsed = t - attempt * ATTEMPT_DURATION
    progress = clamp01(elapsed / FLIGHT_DURATION)
    points, _ = learning_path(attempt, progress)
    pos = bezier(points, smoothstep(progress))
    # Hold the endpoint, then fade out and fade the next attempt in.
    alpha = min(1.0, elapsed / 0.18) if attempt else 1.0
    if attempt < 3 and elapsed > 2.05:
        alpha = clamp01((ATTEMPT_DURATION - elapsed) / 0.35)
    return pos, attempt, progress, alpha


def update_and_draw_stage_learning(surface, t):
    global weights, signal_colour, signal_value
    pos, attempt, local, alpha = learning_pose(t)
    points, _ = learning_path(attempt, local)
    layer = pygame.Surface((BASE_W, BASE_H), pygame.SRCALPHA)
    trail = [bezier(points, i / 79 * smoothstep(local)) for i in range(80)]
    if local > 0:
        pygame.draw.lines(layer, ACCENT, False, trail, 5)
    draw_fly(layer, pos, show_rotation=False, radius=52)
    layer.set_alpha(round(255 * alpha))
    surface.blit(layer, (0, 0))
    text(surface, f"Attempt {attempt + 1} / 4", ARENA.x + 32, ARENA.bottom - 60, DIM, SMALL)
    overall_progress = (attempt + local) / 4.0
    weights = interpolate_weights(BASE_WEIGHTS, GOOD_END_WEIGHTS, overall_progress)
    signal_colour = GOOD
    emit_continuous("reward", min(t, LEARNING_DURATION), pos)


def toggle_fullscreen():
    global screen, fullscreen, windowed_size
    fullscreen = not fullscreen
    if fullscreen:
        windowed_size = screen.get_size()
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode(windowed_size, pygame.RESIZABLE)

# ----------------------------
# Rendering to arbitrary window size
# ----------------------------

canvas = pygame.Surface((BASE_W, BASE_H))

def render_frame(dt):
    canvas.fill(BG)

    if mode == "good":
        update_stage_good(stage_time)
        draw_arena(canvas, show_main_fly=True, show_rotation=True)

    elif mode == "bad":
        update_stage_bad(stage_time)
        draw_arena(canvas, show_main_fly=True, show_rotation=True)

    elif mode == "reward":
        update_stage_reward(stage_time)
        draw_arena(canvas, show_main_fly=True, show_rotation=True)

    elif mode == "punish":
        update_stage_punish(stage_time)
        draw_arena(canvas, show_main_fly=True, show_rotation=True)

    elif mode == "strengthen":
        update_stage_strengthen(stage_time)
        draw_arena(canvas, show_main_fly=True, show_rotation=True)

    elif mode == "weaken":
        update_stage_weaken(stage_time)
        draw_arena(canvas, show_main_fly=True, show_rotation=True)

    elif mode == "learning":
        # Draw only arena border + tree, then the moving fly path.
        pygame.draw.rect(canvas, BORDER, ARENA, 2)
        draw_tree(canvas, tree_pos)
        update_and_draw_stage_learning(canvas, stage_time)

    else:
        draw_arena(canvas, show_main_fly=True, show_rotation=True)

    draw_brain(canvas)
    update_and_draw_particles(canvas, dt)
    draw_brain_labels(canvas)

    # Letterbox while preserving authored aspect ratio.
    sw, sh = screen.get_size()
    scale = min(sw / BASE_W, sh / BASE_H)
    rw, rh = max(1, int(BASE_W * scale)), max(1, int(BASE_H * scale))
    scaled = pygame.transform.smoothscale(canvas, (rw, rh))
    x = (sw - rw) // 2
    y = (sh - rh) // 2

    screen.fill(BG)
    screen.blit(scaled, (x, y))
    pygame.display.flip()

# ----------------------------
# Internal tests
# ----------------------------

def run_self_test():
    global stage_time, current_heading, weights

    # Stage 1 resets correctly.
    start_stage("good")
    assert abs(current_heading - INITIAL_HEADING) < 1e-9
    assert weights == BASE_WEIGHTS

    # Stage 1 endpoint is stable.
    update_stage_good(GOOD_DURATION + 10.0)
    assert abs(current_heading - GOOD_END_HEADING) < 1e-9
    assert all(abs(a-b) < 1e-9 for a,b in zip(weights, GOOD_END_WEIGHTS))

    # Stage 2 starts exactly from stage 1 endpoint.
    start_stage("bad")
    assert abs(current_heading - GOOD_END_HEADING) < 1e-9
    assert all(abs(a-b) < 1e-9 for a,b in zip(weights, GOOD_END_WEIGHTS))

    # Stage 2 endpoint remains fixed no matter how long it runs.
    update_stage_bad(BAD_DURATION + 1.0)
    h1 = current_heading
    w1 = weights[:]
    update_stage_bad(BAD_DURATION + 20.0)
    assert abs(current_heading - h1) < 1e-9
    assert all(abs(a-b) < 1e-9 for a,b in zip(weights, w1))

    # Going back to stage 1 resets original pose.
    start_stage("good")
    assert abs(current_heading - INITIAL_HEADING) < 1e-9
    assert weights == BASE_WEIGHTS

    # Stage 7 starts independently.
    start_stage("learning")
    assert mode == "learning"
    assert abs(current_heading - INITIAL_HEADING) < 1e-9
    assert weights == BASE_WEIGHTS

    # Width mapping is monotonic and smooth.
    vals = [active_width(x/100.0) for x in range(20, 121)]
    assert all(b > a for a,b in zip(vals, vals[1:]))

    # Finished stages stop emitting and their particles drain completely.
    for stage, duration in [("good", 3), ("bad", 3), ("strengthen", 2.5),
                            ("weaken", 2.5), ("learning", LEARNING_DURATION)]:
        start_stage(stage)
        stage_time = duration + 2
        render_frame(0)
        assert not floating_numbers and not pulses, stage

    # Rendering a paused frame must not change any pixels.
    start_stage("learning")
    stage_time = 1.2
    render_frame(0)
    before = pygame.image.tobytes(canvas, "RGB")
    render_frame(0)
    assert before == pygame.image.tobytes(canvas, "RGB")

    # Sampling the same time at different frame rates gives the same image.
    frames = []
    for fps in (15, 60):
        start_stage("learning")
        for frame in range(1, fps * 3 + 1):
            stage_time = frame / fps
            render_frame(1 / fps)
        frames.append(pygame.image.tobytes(canvas, "RGB"))
    assert frames[0] == frames[1]
    stage_time = LEARNING_DURATION
    render_frame(0)
    assert weights == GOOD_END_WEIGHTS
    assert learning_pose(LEARNING_DURATION)[0] == pygame.Vector2(tree_pos.x, tree_pos.y + 18)

    print("SELF TEST PASS")
    print(f"GOOD_END_HEADING={GOOD_END_HEADING:.3f}")
    print(f"BAD_END_HEADING={BAD_END_HEADING:.3f}")


def export_preview():
    """Render all stages without opening a window; no extra dependencies."""
    global stage_time
    os.makedirs("artifacts/learning-visualizer-matched", exist_ok=True)
    sheet = pygame.Surface((1600, 900))
    for i, stage in enumerate(("idle", "good", "bad", "reward", "punish", "strengthen", "weaken", "learning")):
        start_stage(stage)
        stage_time = 1.2 if stage != "learning" else LEARNING_DURATION
        render_frame(0)
        tile = pygame.transform.smoothscale(canvas, (400, 225))
        sheet.blit(tile, ((i % 4) * 400, (i // 4) * 450 + 40))
        text(sheet, stage, (i % 4) * 400 + 12, (i // 4) * 450 + 8)
    pygame.image.save(sheet, "artifacts/learning-visualizer-matched/stages.png")
    start_stage("good")
    stage_time = 1.2
    render_frame(0)
    pygame.image.save(canvas, "artifacts/learning-visualizer-matched/detail.png")


# ----------------------------
# Main loop
# ----------------------------

def main():
    global stage_time, paused, fullscreen, screen

    if "--preview" in os.sys.argv:
        export_preview()
        pygame.quit()
        return

    if "--self-test" in os.sys.argv:
        run_self_test()
        pygame.quit()
        return

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_F11:
                    toggle_fullscreen()

                elif event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_r:
                    start_stage("idle")

                elif event.key == pygame.K_1:
                    start_stage("good")

                elif event.key == pygame.K_2:
                    start_stage("bad")

                elif event.key == pygame.K_3:
                    start_stage("reward")

                elif event.key == pygame.K_4:
                    start_stage("punish")

                elif event.key == pygame.K_5:
                    start_stage("strengthen")

                elif event.key == pygame.K_6:
                    start_stage("weaken")

                elif event.key == pygame.K_7:
                    start_stage("learning")

            elif event.type == pygame.VIDEORESIZE and not fullscreen:
                screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)

        if not paused:
            stage_time += dt

        render_frame(0.0 if paused else dt)

    pygame.quit()


if __name__ == "__main__":
    main()
