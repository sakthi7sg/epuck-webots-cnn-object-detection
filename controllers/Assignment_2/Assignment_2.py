"""E-puck block-search controller (Webots, Python).

Brief: find a red, a blue and a green block (each 0.3 x 0.3 x 0.1 m) placed at
random, print "I see <colour>" on the FIRST encounter of each colour, and after
every response print a summary of which colours have been encountered.

The e-puck node must have  supervisor TRUE  so the controller can move (or, if
they are missing from the world, create) the three blocks.
"""
import random
import math
from controller import Supervisor, Camera

TS = 64                          # control time step (ms)
SPEED = 4.0                      # base wheel speed (rad/s); e-puck maximum is 6.28
OBSTACLE = 80                    # proximity-sensor value meaning "something is close"
ARENA_X, ARENA_Y = 0.37, 0.19    # arena centre (m): set to match the provided world
ARENA_HALF = 0.65                # blocks are placed within +/- this distance of the centre
SEED = None                      # use an integer for a repeatable block layout
COLOURS = {"red": (1, 0, 0), "blue": (0, 0, 1), "green": (0, 1, 0)}
order = []                       # colours in the order they were first encountered


# 1. PLACE THE THREE BLOCKS AT RANDOM
def create_blocks(robot):
    # A block already in the world (DEF RED_BLOCK, BLUE_BLOCK or GREEN_BLOCK) is
    # moved; a missing one is created, so there are always exactly three blocks.
    start = robot.getSelf().getField("translation").getSFVec3f()[:2]
    children = robot.getRoot().getField("children")
    placed = []
    for colour, rgb in COLOURS.items():
        name = colour.upper() + "_BLOCK"
        for _ in range(1000):
            x = ARENA_X + random.uniform(-ARENA_HALF, ARENA_HALF)
            y = ARENA_Y + random.uniform(-ARENA_HALF, ARENA_HALF)
            p = (x, y)
            if math.dist(p, start) < 0.5:                  # keep clear of the robot
                continue
            if any(math.dist(p, q) < 0.55 for q in placed):  # keep blocks apart
                continue
            node = robot.getFromDef(name)
            if node is not None:
                node.getField("translation").setSFVec3f([x, y, 0.05])
            else:
                r, g, b = rgb
                block = f"""
                DEF {name} Solid {{
                  translation {x} {y} 0.05
                  name "{name}"
                  children [
                    Shape {{
                      appearance PBRAppearance {{ baseColor {r} {g} {b} }}
                      geometry Box {{ size 0.3 0.3 0.1 }}
                    }}
                  ]
                  boundingObject Box {{ size 0.3 0.3 0.1 }}
                }}
                """
                children.importMFNodeFromString(-1, block)
                if robot.getFromDef(name) is None:
                    print("WARNING: could not create the " + colour + " block")
            placed.append(p)
            print("%s block at x = %.2f m, y = %.2f m" % (colour.capitalize(), x, y))
            break
        else:
            print("WARNING: could not find a free position for the " + colour + " block")


# 2. IDENTIFY COLOUR
def classify(r, g, b):
    if r > 80 and g < 0.3 * r and b < 0.3 * r: return "red"
    if g > 80 and r < 0.3 * g and b < 0.3 * g: return "green"
    if b > 80 and r < 0.3 * b and g < 0.3 * b: return "blue"
    return None


def scan(image, w, h):
    count = {c: 0 for c in COLOURS}     # pixels of each colour in the whole image
    sx = {c: 0 for c in COLOURS}        # sum of their x positions (for the centroid)
    centre = {c: 0 for c in COLOURS}    # pixels inside the 17 x 13 window at the centre
    for x in range(w):
        for y in range(h):
            r = Camera.imageGetRed(image, w, x, y)
            g = Camera.imageGetGreen(image, w, x, y)
            b = Camera.imageGetBlue(image, w, x, y)
            c = classify(r, g, b)
            if c:
                count[c] += 1
                sx[c] += x
                if abs(x - w // 2) <= 8 and abs(y - h // 2) <= 6:
                    centre[c] += 1
    return count, sx, centre


# 3. REPORT THE FIRST ENCOUNTER OF A COLOUR
def report(c, t):
    previous = list(order)
    order.append(c)
    missing = [k for k in COLOURS if k not in order]
    print("\nI see " + c)
    print("Previously encountered:", ", ".join(previous) or "None")
    print("Encountered so far:", ", ".join(order))
    print("Not yet found:", ", ".join(missing) or "None")
    print("Simulation time: %.1f s" % t)


# 4. MAIN ROBOT
def main():
    robot = Supervisor()
    random.seed(SEED)
    create_blocks(robot)

    camera = robot.getDevice("camera")
    camera.enable(TS)
    w, h = camera.getWidth(), camera.getHeight()

    sensors = [robot.getDevice("ps" + str(i)) for i in range(8)]
    for s in sensors:
        s.enable(TS)

    left = robot.getDevice("left wheel motor")
    right = robot.getDevice("right wheel motor")
    for m in (left, right):
        m.setPosition(float("inf"))
        m.setVelocity(0)

    retreat = turn = wander = 0
    direction = 1
    done = False
    threshold = 0.5 * 17 * 13       # half of the 17 x 13 window at the image centre

    while robot.step(TS) != -1:
        if done:                    # all three found: wheels stay stopped
            continue

        ps = [s.getValue() for s in sensors]
        front = any(ps[i] > OBSTACLE for i in (0, 1, 6, 7))
        image = camera.getImage()
        if image is None:
            continue

        count, sx, centre = scan(image, w, h)
        unfound = [c for c in COLOURS if c not in order]

        # First encounter: close to a block whose colour has not been seen before
        if not (retreat or turn) and front and unfound:
            c = max(unfound, key=lambda k: centre[k])
            if centre[c] >= threshold:
                report(c, robot.getTime())
                if not [k for k in COLOURS if k not in order]:
                    print("\nAll three blocks found. Stopping.")
                    left.setVelocity(0)
                    right.setVelocity(0)
                    done = True
                    continue
                retreat, turn = 12, 14
                direction = random.choice([-1, 1])

        # Movement and obstacle avoidance
        if retreat:
            l, r = -SPEED, -SPEED
            retreat -= 1
        elif turn:
            l, r = direction * SPEED, -direction * SPEED
            turn -= 1
        elif any(ps[i] > OBSTACLE for i in (5, 6, 7)):
            l, r = SPEED, -SPEED
            wander = 0
        elif any(ps[i] > OBSTACLE for i in (0, 1, 2)):
            l, r = -SPEED, SPEED
            wander = 0
        else:
            # steer towards the largest block whose colour has not been found yet
            targets = [c for c in unfound if count[c] >= 8]
            if targets:
                c = max(targets, key=lambda k: count[k])
                error = (sx[c] / count[c] - (w - 1) / 2) / (w / 2)
                l = max(-6.28, min(6.28, SPEED * (1 + 0.8 * error)))
                r = max(-6.28, min(6.28, SPEED * (1 - 0.8 * error)))
            elif wander:
                l, r = direction * SPEED, -direction * SPEED
                wander -= 1
            elif random.random() < 0.01:
                direction = random.choice([-1, 1])
                wander = random.randint(5, 15)
                l, r = direction * SPEED, -direction * SPEED
            else:
                l = r = SPEED

        left.setVelocity(l)
        right.setVelocity(r)


if __name__ == "__main__":
    main()