import re
with open('rtl/texture_unit.v', 'r') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if 'reg [3:0]  filter_r, wrap_s_r, wrap_t_r;' in line:
        new_lines.append(line)
        new_lines.append('    reg [3:0]  format_r;\n')
        new_lines.append('    reg [4:0]  texel_size_r;\n')
    elif 'wrap_s_r <= tex_wrap_s;' in line:
        new_lines.append(line)
        new_lines.append('                        format_r <= tex_format;\n')
        new_lines.append('''                        case (tex_format)
                            FMT_R8_UNORM: texel_size_r <= 5'd1;
                            FMT_RG8_UNORM, FMT_R16_FLOAT: texel_size_r <= 5'd2;
                            FMT_RGBA8_UNORM, FMT_RGBA8_SNORM, FMT_R32_FLOAT, FMT_RG16_FLOAT: texel_size_r <= 5'd4;
                            FMT_RGBA16_FLOAT, FMT_RG32_FLOAT: texel_size_r <= 5'd8;
                            FMT_RGBA32_FLOAT: texel_size_r <= 5'd16;
                            default: texel_size_r <= 5'd4;
                        endcase
''')
    elif 'tex_w_r * 4' in line:
        new_lines.append(line.replace('tex_w_r * 4', 'tex_w_r * {11\'b0, texel_size_r}').replace('4\'d4', 'texel_size_r[3:0]'))
    elif 'tex_w_r * tex_h_r * 4' in line:
        new_lines.append(line.replace('tex_w_r * tex_h_r * 4', 'tex_w_r * tex_h_r * {11\'b0, texel_size_r}'))
    elif '* 4;' in line:
        new_lines.append(line.replace('* 4;', '* {27\'b0, texel_size_r};'))
    elif '* 4)' in line:
        new_lines.append(line.replace('* 4)', '* {11\'b0, texel_size_r})'))
    else:
        new_lines.append(line)

with open('rtl/texture_unit.v', 'w') as f:
    f.writelines(new_lines)
