"""Minimal ACL diagnostic — isolate the model load failure."""

import json
import os
import sys
import ctypes

sys.path.insert(0, "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages")
sys.path.insert(0, "/usr/local/Ascend/ascend-toolkit/latest/pyACL/python/site-packages")

import acl

def check(ret, tag):
    if ret != 0:
        print(f"  FAIL [{tag}] → ret={ret} (0x{ret:08X})")
    else:
        print(f"  OK   [{tag}]")

print("=== ACL Diagnostic ===")
print(f"Python: {sys.version}")
print(f"ACL module: {acl.__file__}")

# 1. Version
try:
    ver = acl.get_version()
    print(f"ACL version: {ver}")
except Exception as e:
    print(f"get_version: {e}")

# 2. Hugepages
print("\n--- Hugepages ---")
os.system("grep -i huge /proc/meminfo 2>/dev/null | head -5")

# 3. Init
print("\n--- Init ---")
ret = acl.init("")
check(ret, "acl.init")

if ret != 0:
    print("ABORT: init failed")
    sys.exit(1)

# 4. Device
print("\n--- Device ---")
try:
    count, ret = acl.rt.get_device_count()
    check(ret, f"get_device_count → {count}")
except Exception as e:
    print(f"  get_device_count failed: {e}")
    count = 1

ret = acl.rt.set_device(0)
check(ret, "set_device(0)")

ctx, ret = acl.rt.create_context(0)
check(ret, "create_context(0)")

# 5. Try loading OM with detailed error
print("\n--- Load OM ---")
om_path = "model/yolov8n.om"
if not os.path.isfile(om_path):
    om_path = "/mnt/e61/workspace/dmd/model/yolov8n.om"
    if not os.path.isfile(om_path):
        print(f"  OM file not found!")
    else:
        print(f"  Found: {om_path}")

st = os.stat(om_path)
print(f"  Size: {st.st_size} bytes ({st.st_size/1024/1024:.1f} MB)")

# Try load
model_id, ret = acl.mdl.load_from_file(om_path)
print(f"  acl.mdl.load_from_file → model_id={model_id}, ret={ret} (0x{ret:08X})")

if ret != 0:
    err = acl.get_recent_err_msg()
    print(f"  Error msg: {err}")

    # Try memory-assisted load
    print("\n  Trying mdl.load_from_file_with_mem...")
    try:
        # alloc device mem for model
        model_size = st.st_size
        mem_ptr, ret_mem = acl.rt.malloc(model_size, 0)  # 0 = ACL_MEM_MALLOC_HUGE_FIRST
        check(ret_mem, "rt.malloc for model")
        if ret_mem == 0:
            model_id2, ret2 = acl.mdl.load_from_file_with_mem(om_path, mem_ptr, model_size)
            print(f"  mdl.load_from_file_with_mem → model_id={model_id2}, ret={ret2} (0x{ret2:08X})")
            if ret2 == 0:
                err = acl.get_recent_err_msg()
                print(f"  Error msg: {err}")
    except Exception as e:
        print(f"  load_from_file_with_mem exception: {e}")

# 6. Cleanup
print("\n--- Cleanup ---")
try:
    acl.rt.destroy_context(ctx)
    acl.rt.reset_device(0)
except Exception as e:
    print(f"  destroy context: {e}")
ret = acl.finalize()
check(ret, "acl.finalize")

print("\n=== Done ===")
