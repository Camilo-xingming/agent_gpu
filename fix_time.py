import sys
fname = "rtl/streaming_multiprocessor_v2.v"
with open(fname, "r") as f: lines = f.readlines()
for i, line in enumerate(lines):
    if "if ($time < 10000000 && $time % 100000 == 0) begin" in line:
        lines[i] = "`ifdef SIMULATION
" + line
        j = i + 1
        count = 1
        while j < len(lines) and count > 0:
            if "begin" in lines[j]: count += 1
            if "end" in lines[j]: count -= 1
            j += 1
        lines[j-1] = lines[j-1] + "`endif
"
        break
with open(fname, "w") as f: f.writelines(lines)