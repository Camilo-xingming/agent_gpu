import sys
fname = "rtl/gpu_defines.vh"
with open(fname, "r") as f: lines = f.readlines()
bt = chr(96)
insert_idx = -1
for i, line in enumerate(lines):
    if bt + "include \"memory_config.vh\"" in line:
        insert_idx = i + 1
        break
if insert_idx != -1:
    block = [bt + "ifdef SYNTH_REDUCED
", "    " + bt + "define THREADS_PER_WARP    4
", "    " + bt + "define WARPS_PER_SM       2
", "    " + bt + "define NUM_REGS           16
", bt + "endif
", "
"]
    lines[insert_idx:insert_idx] = block
    with open(fname, "w") as f: f.writelines(lines)
    print("SUCCESS")
else:
    print("NOT FOUND")