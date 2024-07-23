import time

prev_time = None


def start_profiler():
    global prev_time
    prev_time = time.time()


def profiler_checkpoint_blank():
    global prev_time
    current_time = time.time()
    prev_time = current_time


def profiler_checkpoint(description):
    global prev_time
    current_time = time.time()
    print(f"{description}: {current_time - prev_time}")
    prev_time = current_time
