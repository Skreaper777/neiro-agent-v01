# game_with_ai_agent.py — модифицированная версия с записью и имитирующим агентом

import os
os.environ['SDL_VIDEO_WINDOW_POS'] = "1800,100"

import pygame, sys, math, json
from collections import deque
import copy

# 1) Конфиг
with open("config.json", encoding="utf-8") as f:
    cfg = json.load(f)

WINDOW_WIDTH     = cfg["window_size"]["width"]
WINDOW_HEIGHT    = cfg["window_size"]["height"]
BG_COLOR         = tuple(cfg["background_color"])
TILESET_PATH     = cfg["tileset_path"]
ADAM_SPRITE_PATH = cfg["adam_sprite_path"]
TILE_W, TILE_H   = cfg["tile_size"]["w"], cfg["tile_size"]["h"]

FOOT_W_RATIO = cfg["foot_collision"]["width_ratio"]
FOOT_H_RATIO = cfg["foot_collision"]["height_ratio"]

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Адам и Агент")

with open("map.json", encoding="utf-8") as f:
    map_data = json.load(f)

MAP_COLS = map_data["width"]
MAP_ROWS = map_data["height"]

layer    = next(l for l in map_data["layers"] if l["type"] == "tilelayer")
map_gids = layer["data"]

tileset    = pygame.image.load(TILESET_PATH).convert_alpha()
ts         = map_data["tilesets"][0]
firstgid   = ts["firstgid"]
cols_ts    = ts["columns"]
tilecount  = ts.get("tilecount", cols_ts * (ts["imageheight"]//TILE_H))

tiles       = {}
coll_shapes = {}

unique_gids = {gid for gid in map_gids if gid > 0}
for gid in unique_gids:
    idx = gid - firstgid
    if 0 <= idx < tilecount:
        x0 = (idx % cols_ts) * TILE_W
        y0 = (idx // cols_ts) * TILE_H
        tiles[gid] = tileset.subsurface(pygame.Rect(x0, y0, TILE_W, TILE_H))

for tile in ts.get("tiles", []):
    gid = firstgid + tile["id"]
    if gid not in unique_gids:
        continue
    objs = tile.get("objectgroup", {}).get("objects", [])
    shapes = [pygame.Rect(o["x"], o["y"], o["width"], o["height"]) for o in objs]
    if shapes:
        coll_shapes[gid] = shapes

# Спрайты Адама
FRAME_W, FRAME_H = 16, 32
SCALE            = 2
SW, SH           = FRAME_W * SCALE, FRAME_H * SCALE
ANIM_SPEED       = 0.12
speed            = 2

adam_sheet = pygame.image.load(ADAM_SPRITE_PATH).convert_alpha()
def slice_adam(sheet):
    frames = [[] for _ in range(4)]
    for d in range(4):
        for i in range(6):
            surf = sheet.subsurface(pygame.Rect((d*6 + i)*FRAME_W, 0, FRAME_W, FRAME_H))
            frames[d].append(pygame.transform.scale(surf, (SW, SH)))
    return frames

adam_frames = slice_adam(adam_sheet)

FOOT_W = SW * FOOT_W_RATIO
FOOT_H = SH * FOOT_H_RATIO

# Начальные координаты
start_col = MAP_COLS // 2
adam_x = start_col * TILE_W + TILE_W//2
adam_y = (MAP_ROWS - 1) * TILE_H

agent_x = adam_x + TILE_W * 2
agent_y = adam_y

frame_index = 0
anim_timer  = 0
current_dir = 3

# --- Новое: запись и имитатор ---
recording = False
repeating = False
demo_buffer = []

def extract_vision(cx, cy):
    result = []
    for dy in range(-2, 3):
        row = []
        for dx in range(-2, 3):
            tx, ty = cx + dx, cy + dy
            if 0 <= tx < MAP_COLS and 0 <= ty < MAP_ROWS:
                gid = map_gids[ty * MAP_COLS + tx]
            else:
                gid = 0
            row.append(gid)
        result.append(row)
    return result

def get_action_from_keys(keys):
    if keys[pygame.K_LEFT]:  return "left"
    if keys[pygame.K_RIGHT]: return "right"
    if keys[pygame.K_UP]:    return "up"
    if keys[pygame.K_DOWN]:  return "down"
    return None

def find_most_similar_action(state, buffer):
    best = None
    min_dist = float('inf')
    for record in buffer:
        s2 = record["state"]
        dist = sum(abs(a - b) for r1, r2 in zip(state, s2) for a, b in zip(r1, r2))
        if dist < min_dist:
            min_dist = dist
            best = record["action"]
    return best

clock = pygame.time.Clock()
running = True

while running:
    dt = clock.tick(60) / 1000
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_r:
                recording = not recording
                print("[REC]" if recording else "[STOP]")
            if ev.key == pygame.K_t:
                repeating = not repeating
                print("[REPEAT ON]" if repeating else "[REPEAT OFF]")

    # Управление Адамом
    keys = pygame.key.get_pressed()
    dx = dy = 0
    if keys[pygame.K_LEFT]:   dx -= speed
    if keys[pygame.K_RIGHT]:  dx += speed
    if keys[pygame.K_UP]:     dy -= speed
    if keys[pygame.K_DOWN]:   dy += speed
    if dx and dy:
        n = math.hypot(dx, dy)
        dx, dy = dx/n*speed, dy/n*speed
    moving = dx or dy

    if dx>0:     current_dir=0
    elif dx<0:   current_dir=2
    elif dy<0:   current_dir=1
    elif dy>0:   current_dir=3

    foot_rect = pygame.Rect(
        adam_x + dx - FOOT_W/2,
        adam_y + dy - FOOT_H,
        FOOT_W, FOOT_H
    )

    collision = False
    c0 = max(0, foot_rect.left   // TILE_W)
    c1 = min(MAP_COLS, foot_rect.right  // TILE_W + 1)
    r0 = max(0, foot_rect.top    // TILE_H)
    r1 = min(MAP_ROWS, foot_rect.bottom // TILE_H + 1)

    for r in range(r0, r1):
        for c in range(c0, c1):
            gid = map_gids[r*MAP_COLS + c]
            if gid == 0: continue
            shapes = coll_shapes.get(gid)
            if shapes:
                for s in shapes:
                    abs_r = pygame.Rect(c*TILE_W + s.x, r*TILE_H + s.y, s.w, s.h)
                    if foot_rect.colliderect(abs_r):
                        collision = True
                        break
            else:
                abs_r = pygame.Rect(c*TILE_W, r*TILE_H, TILE_W, TILE_H)
                if foot_rect.colliderect(abs_r):
                    collision = True
            if collision: break
        if collision: break

    if not collision:
        adam_x += dx
        adam_y += dy

    if recording:
        col = int(adam_x // TILE_W)
        row = int(adam_y // TILE_H)
        state = extract_vision(col, row)
        action = get_action_from_keys(keys)
        if action:
            demo_buffer.append({"state": state, "action": action})

    # Обновление анимации
    if moving:
        anim_timer += ANIM_SPEED
        if anim_timer >= 1:
            anim_timer = 0
            frame_index = (frame_index + 1) % 6
    else:
        frame_index = 0

    # Агент-имитатор
    if repeating and demo_buffer:
        col = int(agent_x // TILE_W)
        row = int(agent_y // TILE_H)
        state = extract_vision(col, row)
        action = find_most_similar_action(state, demo_buffer)
        ax = ay = 0
        if action == "left":  ax = -speed
        if action == "right": ax = speed
        if action == "up":    ay = -speed
        if action == "down":  ay = speed
        agent_x += ax
        agent_y += ay

    # Отрисовка
    screen.fill(BG_COLOR)

    for r in range(MAP_ROWS):
        for c in range(MAP_COLS):
            gid = map_gids[r*MAP_COLS + c]
            img = tiles.get(gid)
            if img:
                screen.blit(img, (c*TILE_W, r*TILE_H))

    frame = adam_frames[current_dir][frame_index]
    screen.blit(frame, (adam_x - SW/2, adam_y - SH))

    pygame.draw.rect(screen, (255, 0, 0), (agent_x - SW/2, agent_y - SH, SW, SH), 2)
    pygame.display.flip()

pygame.quit()
sys.exit()
