import sys
fname = "rtl/wgmma.v"
with open(fname, "r") as f: content = f.read()
bt = chr(96)
start8 = "module fp8_mma_unit"
start6 = "module fp6_mma_unit"
marker = "for (m = 0; m < M; m = m + 1) begin"
end_m = "valid_out <= 1" + chr(39) + "b1;"
if start8 in content and start6 in content:
    parts = content.split(start8)
    parts2 = parts[1].split(start6)
    fp8_body = parts2[0]
    fp6_body = parts2[1]
    l_start8 = fp8_body.find(marker)
    l_end8 = fp8_body.find(end_m, l_start8) + len(end_m)
    orig8 = fp8_body[l_start8:l_end8]
    new8 = bt + "ifndef SYNTHESIS
            " + orig8 + "
            " + bt + "else
            matrix_d <= 0; valid_out <= 1" + chr(39) + "b1;
            " + bt + "endif"
    fp8_body = fp8_body.replace(orig8, new8)
    l_start6 = fp6_body.find(marker)
    l_end6 = fp6_body.find(end_m, l_start6) + len(end_m)
    orig6 = fp6_body[l_start6:l_end6]
    new6 = bt + "ifndef SYNTHESIS
            " + orig6 + "
            " + bt + "else
            matrix_d <= 0; valid_out <= 1" + chr(39) + "b1;
            " + bt + "endif"
    fp6_body = fp6_body.replace(orig6, new6)
    new_content = parts[0] + start8 + fp8_body + start6 + fp6_body
    with open(fname, "w") as f: f.write(new_content)
    print("SUCCESS")
else:
    print("FAIL")