import os
os.environ['SDL_VIDEO_WINDOW_POS'] = "1800,100"

import pygame, sys, math, json

# 1) Конфиг
with open("config.json", encoding="utf-8") as f:
    cfg = json.load(f)

WINDOW_WIDTH     = cfg["window_size"]["width"]
WINDOW_HEIGHT    = cfg["window_size"]["height"]
SIDE_PANEL_WIDTH = cfg.get("side_panel_width", 200)
BG_COLOR         = tuple(cfg["background_color"])
TILESET_PATH     = cfg["tileset_path"]
ADAM_SPRITE_PATH = cfg["adam_sprite_path"]
TILE_W, TILE_H   = cfg["tile_size"]["w"], cfg["tile_size"]["h"]

# Новые настройки «ступней»
FOOT_W_RATIO = cfg["foot_collision"]["width_ratio"]
FOOT_H_RATIO = cfg["foot_collision"]["height_ratio"]
ADAM_COL = cfg["adam_start"]["col"]
ADAM_ROW = cfg["adam_start"]["row"]
AGENT_COL = cfg["agent_start"]["col"]
AGENT_ROW = cfg["agent_start"]["row"]

# 2) Инициализация Pygame и окно
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH + SIDE_PANEL_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Комната и Адам")
font = pygame.font.SysFont(None, 24)

# 3) Загрузка и парсинг map.json (включая tileset с objectgroup)
with open("map.json", encoding="utf-8") as f:
    map_data = json.load(f)

MAP_COLS = map_data["width"]
MAP_ROWS = map_data["height"]
if map_data["tilewidth"] != TILE_W or map_data["tileheight"] != TILE_H:
    raise RuntimeError("Несовпадение размеров тайлов")

layer    = next(l for l in map_data["layers"] if l["type"] == "tilelayer")
map_gids = layer["data"]

# 4) Вырезаем все тайлы и парсим collision‑shapes
tileset    = pygame.image.load(TILESET_PATH).convert_alpha()
ts         = map_data["tilesets"][0]
firstgid   = ts["firstgid"]
cols_ts    = ts["columns"]
tilecount  = ts.get("tilecount", cols_ts * (ts["imageheight"]//TILE_H))

tiles       = {}
coll_shapes = {}
zone_rects  = {}

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

# Обработка зон
for layer in map_data["layers"]:
    if layer["type"] == "objectgroup" and layer["name"] == "zones":
        for obj in layer["objects"]:
            for prop in obj.get("properties", []):
                if prop["name"] == "zone":
                    zone_name = prop["value"]
                    rect = pygame.Rect(obj["x"], obj["y"], obj["width"], obj["height"])
                    zone_rects[zone_name] = rect

# 5) Спрайты Адама
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

# **Зона столкновений «ступней»**
FOOT_W = SW * FOOT_W_RATIO
FOOT_H = SH * FOOT_H_RATIO

adam_x = ADAM_COL * TILE_W + TILE_W//2
adam_y = ADAM_ROW * TILE_H + TILE_H//2
agent_x = AGENT_COL * TILE_W + TILE_W//2
agent_y = AGENT_ROW * TILE_H + TILE_H//2

frame_index = 0
anim_timer  = 0
current_dir = 3

clock = pygame.time.Clock()
running = True

# === Агентное состояние ===
agent_state = {
    "time_indoors": 0.0,
    "time_outdoors": 0.0,
    "want_to_go_home": 0.0,
    "want_to_explore": 0.0,
}

# === Вспомогательные ===
def is_in_zone(name, x, y):
    z = zone_rects.get(name)
    return z and z.collidepoint(x, y)

# === Игровой цикл ===
while running:
    dt = clock.tick(60) / 1000
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False

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

    # === Обновление мотиваций ===
    if is_in_zone("home", agent_x, agent_y - SH/2):
        agent_state["time_indoors"] += dt
        agent_state["time_outdoors"] = 0
        agent_state["want_to_explore"] += dt * 0.5
        agent_state["want_to_go_home"] = 0
    else:
        agent_state["time_outdoors"] += dt
        agent_state["time_indoors"] = 0
        agent_state["want_to_explore"] = 0
        agent_state["want_to_go_home"] += dt * 0.5

    # === Агент выбирает направление ===
    directions = {
        (1, 0): "right",
        (-1, 0): "left",
        (0, -1): "up",
        (0, 1): "down"
    }
    walkable_dirs = []
    for (dx, dy), _ in directions.items():
        ax = dx * speed
        ay = dy * speed
        test_rect = pygame.Rect(agent_x + ax - FOOT_W/2, agent_y + ay - FOOT_H, FOOT_W, FOOT_H)

        a_c0 = max(0, test_rect.left   // TILE_W)
        a_c1 = min(MAP_COLS, test_rect.right  // TILE_W + 1)
        a_r0 = max(0, test_rect.top    // TILE_H)
        a_r1 = min(MAP_ROWS, test_rect.bottom // TILE_H + 1)

        collision = False
        for r in range(a_r0, a_r1):
            for c in range(a_c0, a_c1):
                gid = map_gids[r*MAP_COLS + c]
                if gid == 0: continue
                shapes = coll_shapes.get(gid)
                if shapes:
                    for s in shapes:
                        abs_r = pygame.Rect(c*TILE_W + s.x, r*TILE_H + s.y, s.w, s.h)
                        if test_rect.colliderect(abs_r):
                            collision = True
                            break
                else:
                    abs_r = pygame.Rect(c*TILE_W, r*TILE_H, TILE_W, TILE_H)
                    if test_rect.colliderect(abs_r):
                        collision = True
                if collision: break
            if collision: break

        if not collision:
            walkable_dirs.append((dx, dy))

    move_dir = None
    if agent_state["want_to_explore"] > agent_state["want_to_go_home"]:
        if walkable_dirs:
            move_dir = walkable_dirs[0]
    elif agent_state["want_to_go_home"] > 0:
        if walkable_dirs:
            move_dir = walkable_dirs[-1]

    if move_dir:
        agent_x += move_dir[0] * speed
        agent_y += move_dir[1] * speed

    if moving:
        anim_timer += ANIM_SPEED
        if anim_timer >= 1:
            anim_timer = 0
            frame_index = (frame_index + 1) % 6
    else:
        frame_index = 0

    # отрисовка
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

    # визуализация зоны home
    home_rect = zone_rects.get("home")
    if home_rect:
        surf = pygame.Surface((home_rect.w, home_rect.h), pygame.SRCALPHA)
        surf.fill((0, 0, 255, 60))
        screen.blit(surf, (home_rect.x, home_rect.y))
        label = font.render("home", True, (255, 255, 255))
        screen.blit(label, (home_rect.x + 4, home_rect.y + 4))

    # правая панель
    panel_x = WINDOW_WIDTH + 10
    lines = [
        f"indoors: {agent_state['time_indoors']:.1f}s",
        f"outdoors: {agent_state['time_outdoors']:.1f}s",
        f"go home: {agent_state['want_to_go_home']:.1f}",
        f"explore: {agent_state['want_to_explore']:.1f}"
    ]
    for i, line in enumerate(lines):
        text = font.render(line, True, (255, 255, 255))
        screen.blit(text, (panel_x, 30 + i * 30))

    pygame.display.flip()

pygame.quit()
sys.exit()