import sys
fname = "rtl/memory_config.vh"
with open(fname, "r") as f: content = f.read()
bt = chr(96)
header = """
define L1D_SIZE_KB         1
    define L1D_LINE_SIZE       128
    define L1I_WAYS            1
    define L2_SIZE_KB          32
    define L2_WAYS             1
    define SMEM_SIZE_KB        1
    define RF_SIZE_KB          16
    define TEX_CACHE_SIZE_KB   1
    define TEX_LINE_SIZE       128
    else
""".replace("endif // MEMORY_CONFIG_VH
    last_endif = content.rfind(bt + "endif")
    if last_endif != -1:
        actual_content = content[:last_endif]
        pre_config, config_body = actual_content.split(marker, 1)
        new_content = pre_config + header + marker + config_body + footer + content[last_endif:]
        with open(fname, "w") as f: f.write(new_content)
        print("SUCCESS")
else:
    print("MARKER NOT FOUND")