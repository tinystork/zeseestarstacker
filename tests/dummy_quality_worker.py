def dummy_worker(data):
    import time

    time.sleep(0.05)
    return {"snr": 1.0, "stars": 1.0}, None, 10
