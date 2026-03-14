import sys
fname = "rtl/memory_config.vh"
with open(fname, "r") as f: content = f.read()
bt = chr(96)
content = content.replace(bt + "ifndef SYNTH_REDUCED
", "")
content = content.replace(bt + "endif

// Derived L1D parameters", "// Derived L1D parameters")
start_marker = "// L1 Data Cache Configuration"
end_marker = "//============================================================================
// Constant Memory"
if start_marker in content and end_marker in content:
    parts = content.split(start_marker)
    parts2 = parts[1].split(end_marker)
    new_content = parts[0] + bt + "ifndef SYNTH_REDUCED
" + start_marker + parts2[0] + bt + "endif

" + end_marker + parts2[1]
    with open(fname, "w") as f: f.write(new_content)
    print("SUCCESS")
else:
    print("MARKERS NOT FOUND")