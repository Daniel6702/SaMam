import torch
import time
import os

device = "cuda:0"
tensors = []

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

chunk_sizes_mb = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]

try:
    while True:
        allocated_any = False

        for mb in chunk_sizes_mb:
            try:
                num_bytes = mb * 1024 * 1024
                t = torch.empty(num_bytes, dtype=torch.uint8, device=device)
                t.fill_(1)
                tensors.append(t)
                allocated_any = True
                print(f"Allocated +{mb} MiB, chunks={len(tensors)}")
                time.sleep(0.02)

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue

        if not allocated_any:
            print("At practical limit; retrying...")
            time.sleep(5)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    print("Releasing GPU memory...")
    del tensors
    torch.cuda.empty_cache()
    print("Done.")