import re

# 1. Patch texture_unit.v
with open('rtl/texture_unit.v', 'r') as f:
    rtl_content = f.read()

# Fix calc_2d_addr signature
rtl_content = rtl_content.replace('input [3:0]  bpp;', 'input [4:0]  bpp;')

# Fix TEX_1D address calc
rtl_content = rtl_content.replace(
    "wrap_coord(int_s, tex_w_r, wrap_s_r);\n                            num_texels <= 3'd1;",
    "wrap_coord(int_s, tex_w_r, wrap_s_r) * {27'b0, texel_size_r};\n                            num_texels <= 3'd1;"
)

# Fix calc_2d_addr invocation
rtl_content = rtl_content.replace('texel_size_r[3:0]', 'texel_size_r')

with open('rtl/texture_unit.v', 'w') as f:
    f.write(rtl_content)

# 2. Patch tb_texture_unit.v
with open('tb/tb_texture_unit.v', 'r') as f:
    tb_content = f.read()

# Fix TEX_1D tests to expect * 4
tb_content = tb_content.replace('32\'h00000005}, "TEX 1D s=5"', '32\'h00000014}, "TEX 1D s=5"')
tb_content = tb_content.replace('32\'h00000064}, "TEX 1D s=100"', '32\'h00000190}, "TEX 1D s=100"')
# s=255 -> 255*4=1020=0x03FC. Mem responder gives addr[7:0] for R, addr[15:8] for G
tb_content = re.sub(r'check_result\(\{32\'h000000FF,\s*32\'h000000AA,\s*32\'h00000000,\s*32\'h000000FF\},\s*"TEX 1D s=255 \(last\)"\)', 
                    r'check_result({32\'h000000FF, 32\'h000000AA, 32\'h00000003, 32\'h000000FC}, "TEX 1D s=255 (last)")', tb_content)

tb_content = re.sub(r'check_result\(\{32\'h000000FF,\s*32\'h000000AA,\s*32\'h00000000,\s*32\'h000000FF\},\s*"TEX 1D clamp\(300\)=255"\)', 
                    r'check_result({32\'h000000FF, 32\'h000000AA, 32\'h00000003, 32\'h000000FC}, "TEX 1D clamp(300)=255")', tb_content)

tb_content = tb_content.replace('32\'h00000004}, "TEX 1D repeat(260%256)=4"', '32\'h00000010}, "TEX 1D repeat(260%256)=4"')
tb_content = tb_content.replace('32\'h00000001}, "TEX 1D repeat(257%256)=1"', '32\'h00000004}, "TEX 1D repeat(257%256)=1"')

# 211*4 = 844 = 0x34C. R=4C, G=03.
tb_content = re.sub(r'check_result\(\{32\'h000000FF,\s*32\'h000000AA,\s*32\'h00000000,\s*32\'h000000D3\},\s*"TEX 1D mirror\(300\)=211"\)', 
                    r'check_result({32\'h000000FF, 32\'h000000AA, 32\'h00000003, 32\'h0000004C}, "TEX 1D mirror(300)=211")', tb_content)

tb_content = re.sub(r'check_result\(\{32\'h000000FF,\s*32\'h000000AA,\s*32\'h00000000,\s*32\'h000000FF\},\s*"TEX 1D mirror\(256\)=255"\)', 
                    r'check_result({32\'h000000FF, 32\'h000000AA, 32\'h00000003, 32\'h000000FC}, "TEX 1D mirror(256)=255")', tb_content)

# Section 11 back-to-back TEX 1D
tb_content = tb_content.replace('32\'h0000000A, "back-to-back #1 s=10 R=0x0A"', '32\'h00000028, "back-to-back #1 s=10 R=0x28"')
tb_content = tb_content.replace('32\'h00000014, "back-to-back #2 s=20 R=0x14"', '32\'h00000050, "back-to-back #2 s=20 R=0x50"')
tb_content = tb_content.replace('32\'h0000001E, "back-to-back #3 s=30 R=0x1E"', '32\'h00000078, "back-to-back #3 s=30 R=0x78"')
tb_content = tb_content.replace('32\'h00000028, "back-to-back #4 s=40 R=0x28"', '32\'h000000A0, "back-to-back #4 s=40 R=0xA0"')
tb_content = tb_content.replace('32\'h00000032, "back-to-back #5 s=50 R=0x32"', '32\'h000000C8, "back-to-back #5 s=50 R=0xC8"')

tb_content = tb_content.replace('32\'h00000007, "back-to-back TEX after TXQ R=0x07"', '32\'h0000001C, "back-to-back TEX after TXQ R=0x1C"')
tb_content = tb_content.replace('32\'h00000063, "Post-reset TEX 1D s=99 R=0x63"', '32\'h0000018C, "Post-reset TEX 1D s=99 R=0x18C"')
tb_content = tb_content.replace('32\'h0000007F, "clamp s=127 exact boundary"', '32\'h000001FC, "clamp s=127 exact boundary"')
tb_content = tb_content.replace('32\'h0000007F, "clamp s=128 clamps to 127"', '32\'h000001FC, "clamp s=128 clamps to 127"')
tb_content = tb_content.replace('32\'h0000004D, "Post-unknown-opcode TEX 1D s=77 R=0x4D"', '32\'h00000134, "Post-unknown-opcode TEX 1D s=77 R=0x134"')

# Fix format tests
# R8_UNORM: x=9, y=5. base=0xC0000. pitch=64*1=64. addr = base + 5*64 + 9*1 = 0xC0000 + 320 + 9 = 0xC0149. R=49.
tb_content = tb_content.replace('32\'h00000024, "format R8_UNORM sample"', '32\'h00000049, "format R8_UNORM sample"')

# RGBA16_FLOAT: bpp=8. pitch=64*8=512. addr = base + 5*512 + 9*8 = 0xC0000 + 2560 + 72 = 0xC0A48. R=48.
tb_content = tb_content.replace('32\'h00000024, "format RGBA16_FLOAT sample"', '32\'h00000048, "format RGBA16_FLOAT sample"')

# RGBA32_FLOAT: bpp=16. pitch=64*16=1024. addr = base + 5*1024 + 9*16 = 0xC0000 + 5120 + 144 = 0xC0000 + 0x1400 + 0x90 = 0xC1490. R=90.
tb_content = tb_content.replace('32\'h00000024, "format RGBA32_FLOAT sample"', '32\'h00000090, "format RGBA32_FLOAT sample"')

tb_content = tb_content.replace(
    "tex_filter = 4'h2;",
    "tex_format = 4'h0;\n        tex_filter = 4'h2;"
)

with open('tb/tb_texture_unit.v', 'w') as f:
    f.write(tb_content)
