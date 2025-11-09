import time

# Old method - counting lines with text iteration
start = time.time()
count_old = 0
with open('covertype_dataset.csv', 'r', encoding='utf-8', errors='ignore') as f:
    count_old = sum(1 for _ in f) - 1
old_time = time.time() - start

# New method - binary buffered counting
start = time.time()
count_new = 0
with open('covertype_dataset.csv', 'rb') as f:
    buffer_size = 65536
    buf = f.read(buffer_size)
    while buf:
        count_new += buf.count(b'\n')
        buf = f.read(buffer_size)
count_new = count_new - 1  # Subtract header
new_time = time.time() - start

print(f"Old method: {count_old:,} rows in {old_time:.2f}s")
print(f"New method: {count_new:,} rows in {new_time:.2f}s")
print(f"Speedup: {old_time/new_time:.1f}x faster")
