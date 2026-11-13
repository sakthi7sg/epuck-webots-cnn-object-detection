"""E-puck camera classifier (Webots, Python).

The e-puck turns slowly on the spot and classifies what its camera sees with the
CIFAR-10 CNN trained in cnn/cifar10_cnn.ipynb. Each classified frame is saved as
frames/camera_frame_XXXX.png so it can be uploaded to the notebook (section 7).

Optimisations so the CNN does not disturb the robot's movement:
  * the CNN runs in a background thread, so the 64 ms control loop never waits;
  * only one frame every INFER_EVERY control steps is classified;
  * PyTorch uses a single CPU thread, as a small robot computer would;
  * optional static int8 quantisation (USE_INT8) makes each classification faster.

Setup: put cifar10_cnn.pth (downloaded from the notebook) in this folder and
install PyTorch in the Python that Webots uses:  pip install torch
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor

from controller import Robot

TIME_STEP = 64           # control step (ms)
TURN_SPEED = 0.6         # wheel speed while scanning (rad/s); e-puck maximum is 6.28
INFER_EVERY = 8          # classify one frame every 8 steps (about 0.5 s)
MIN_CONFIDENCE = 0.50    # report a prediction only above this probability
MAX_SAVED_FRAMES = 40    # stop saving images after this many
USE_INT8 = False         # True: static int8 quantisation (faster, about 4x smaller)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "cifar10_cnn.pth")
FRAME_DIR = os.path.join(HERE, "frames")


def load_classifier():
    """Load the trained CNN. Returns (model, torch module) or (None, None) on failure."""
    try:
        import torch
        from cnn_model import CifarCNN
    except ImportError as err:
        print(f"[CNN] PyTorch is not available in Webots' Python ({err}).")
        print("[CNN] Install it with:  pip install torch   then restart Webots.")
        return None, None
    if not os.path.isfile(MODEL_PATH):
        print(f"[CNN] Model file not found: {MODEL_PATH}")
        print("[CNN] Download cifar10_cnn.pth from the notebook and copy it into this folder.")
        return None, None

    torch.set_num_threads(1)
    model = CifarCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    if USE_INT8:
        model = quantise_int8(model, torch)
    print(f"[CNN] Model loaded ({'int8' if USE_INT8 else 'float32'}).")
    return model, torch


def quantise_int8(model, torch):
    """Static post-training int8 quantisation, calibrated on random inputs.

    Calibrating on real CIFAR-10 images (as in the notebook) gives better accuracy;
    random inputs are used here so the robot does not need the dataset.
    """
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    engines = torch.backends.quantized.supported_engines
    torch.backends.quantized.engine = "x86" if "x86" in engines else engines[-1]
    example = torch.randn(1, 3, 32, 32)
    prepared = prepare_fx(model, get_default_qconfig_mapping(torch.backends.quantized.engine), (example,))
    with torch.inference_mode():
        for _ in range(20):
            prepared(torch.randn(16, 3, 32, 32))
    return convert_fx(prepared)


def camera_to_rgb(camera, torch):
    """Webots camera image (BGRA bytes) -> H x W x 3 uint8 RGB tensor."""
    w, h = camera.getWidth(), camera.getHeight()
    bgra = torch.frombuffer(bytearray(camera.getImage()), dtype=torch.uint8).view(h, w, 4)
    return bgra[:, :, [2, 1, 0]].contiguous()


def classify(model, torch, rgb):
    """Run the CNN on one RGB frame. Returns (class name, probability, time in ms)."""
    from cnn_model import CLASSES, preprocess
    start = time.perf_counter()
    with torch.inference_mode():
        probs = torch.softmax(model(preprocess(rgb)), dim=1)[0]
    conf, idx = probs.max(dim=0)
    return CLASSES[int(idx)], float(conf), (time.perf_counter() - start) * 1000


def main():
    robot = Robot()
    camera = robot.getDevice("camera")
    camera.enable(TIME_STEP)
    left, right = robot.getDevice("left wheel motor"), robot.getDevice("right wheel motor")
    for motor in (left, right):
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)

    model, torch = load_classifier()
    os.makedirs(FRAME_DIR, exist_ok=True)
    worker = ThreadPoolExecutor(max_workers=1)    # the CNN runs here, not in the control loop
    pending, pending_frame = None, None
    step, saved, times_ms = 0, 0, []

    while robot.step(TIME_STEP) != -1:
        # Movement is updated every step, whatever the CNN is doing.
        left.setVelocity(-TURN_SPEED)
        right.setVelocity(TURN_SPEED)
        step += 1
        if model is None:
            continue

        # Collect a finished classification without waiting for it.
        if pending is not None and pending.done():
            label, conf, ms = pending.result()
            times_ms.append(ms)
            t = robot.getTime()
            if conf >= MIN_CONFIDENCE:
                print(f"[{t:6.1f} s] I think this is a {label} ({conf*100:.0f}%) "
                      f"| {ms:.1f} ms | frame {pending_frame}")
            else:
                print(f"[{t:6.1f} s] not sure (best guess {label}, {conf*100:.0f}%) | {ms:.1f} ms")
            pending = None

        # Start a new classification every INFER_EVERY steps, if the worker is free.
        if step % INFER_EVERY == 0 and pending is None:
            rgb = camera_to_rgb(camera, torch)
            pending_frame = "-"
            if saved < MAX_SAVED_FRAMES:
                saved += 1
                pending_frame = f"camera_frame_{saved:04d}.png"
                camera.saveImage(os.path.join(FRAME_DIR, pending_frame), 100)
            pending = worker.submit(classify, model, torch, rgb)

        if step % 300 == 0 and times_ms:
            mean = sum(times_ms) / len(times_ms)
            print(f"[CNN] {len(times_ms)} frames | mean {mean:.1f} ms per frame | "
                  f"{mean / (INFER_EVERY * TIME_STEP) * 100:.1f}% of the time between frames")

    worker.shutdown(wait=False)


if __name__ == "__main__":
    main()
