TARGET:
- GPGPU Compute Core compatible with NVIDIA B300 PTX subset
- Focus: Compute Capability 9.x+ (ALU, Control Flow, Shared Memory, Tensor Core)
- Exclusions: Graphics pipeline, Display Engine, Video Codecs

ISA AUTHORITY:
- Public PTX ISA documentation + `.ctx/ISA_SUBSET.md`
- NOT proprietary NVIDIA microarchitecture

PROCESS LOOP (MUST FOLLOW - NO WORKAROUNDS):

1. ASK - Consult BOTH Codex AND Gemini for next MVU (in parallel if possible):
   ```bash
   # Gemini (can run in background):
   gemini -y "RalphGPU: Review COMPLETENESS.md status. What MVU is needed next? Be critical."

   # Codex (non-interactive):
   codex exec "RalphGPU: Review status. What gaps exist? Propose next MVU. DO NOT make code changes."
   ```
   - **MANDATORY**: BOTH tools must be consulted before proceeding
   - **CONSENSUS**: If both agree → proceed to step 2
   - **DISAGREEMENT**: If disagree → re-ask (max 3 rounds); else pick higher-tier item
   - **DOCUMENT**: Record both recommendations in progress.md

2. IMPLEMENT - Codex writes code (NOT the orchestrator):
   ```bash
   cat [relevant files] | codex exec --full-auto "Implement [MVU description]. Output code."
   ```
   - RTL in `rtl/`
   - Python test generator in `tools/`
   - Update FRM in `tools/gpu_simulator.py`

3. REVIEW - Gemini reviews the code:
   ```bash
   cat [new/modified files] | gemini -m gemini-2.5-pro --yolo \
     "Review for PTX compliance and correctness. PASS or FAIL? Be critical."
   ```
   - If FAIL → return to step 2 with Codex fixing issues

4. VERIFY - Run tests:
   ```bash
   python3 tools/rtl_frm_compare.py --all
   ```
   - If PASS → update `.ctx/ARCHIVE.md` and `.ctx/COMPLETENESS.md`
   - If FAIL → return to step 2

5. COMMIT - Push changes to GitHub after each MVU:
   ```bash
   git add -A && git commit -m "MVU-XXX: [description]" && git push
   ```
   - Commit message should reference MVU number and brief description
   - Push immediately after each successful MVU iteration

RULES:
- Functional correctness > performance
- No raw PTX - use Python generators
- FRM required per MVU
- 3 failures → fallback to simpler design

ISSUE TRACKING (MUST DO):
- After each ASK step, record ALL issues from Codex/Gemini in `.ctx/STATUS.md`
- Mark each issue: ✅ FIXED / ❌ PENDING / ⚠️ PARTIAL
- Do NOT declare MVU complete until ALL issues are addressed
- Continue PROCESS LOOP until no PENDING issues remain

CONTINUOUS EXECUTION (MANDATORY - NO EXCEPTIONS):
- This is an INFINITE LOOP - execute until DONE CONDITIONS met
- After completing one MVU, IMMEDIATELY start next PROCESS LOOP
- DO NOT wait for user input between iterations
- DO NOT ask "should I continue?" or similar questions
- Keep iterating automatically until DONE CONDITIONS met
- Report BOTH completed AND pending items after each cycle

DONE CONDITIONS:
- All Tier 1 & 2 in `.ctx/COMPLETENESS.md` marked DONE (not PARTIAL)
- 100% regression pass
- Codex and Gemini agree: no remaining MVUs
- Zero ❌ PENDING issues in `.ctx/STATUS.md`

FINAL: Output `<promise>DONE</promise>` only when all conditions met.
