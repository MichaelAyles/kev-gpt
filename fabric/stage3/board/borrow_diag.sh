#!/usr/bin/env bash
# Borrow the Kria for one diagnostic run, then restore the live demo.
# Restore runs via trap, so an abort mid-run still puts prod back.
set -u
KRIA=${KRIA:?set KRIA=user@host for the Kria, e.g. export KRIA=ubuntu@kria.local}
BIT=/tmp/kevbuild/nc3_first/mamba_pipe.bin
PACK=/tmp/kevbuild/diag_nc1

BORROWED=0          # only restore if we actually took the board
restore() {
  [ "$BORROWED" = "0" ] && { echo "--- nothing borrowed, no restore ---"; return; }
  echo "--- RESTORE ---"
  ssh $KRIA 'bash ~/RESTORE_LIVE_DEMO.sh >/dev/null 2>&1; sleep 6; \
             ss -tln | grep -q :9099 && echo DAEMON_UP || echo DAEMON_DOWN'
  # daemon needs ~25 s to load weights; restart the server AFTER that or it
  # serves empty completions (known model-swap reconnect quirk).
  sleep 30
  ssh precision 'systemctl --user restart kevin-chat-server; \
                 systemctl --user start kevin-rotation; \
                 cd ~/kev-gpt/webchat/demo && cp client.html.prebanner client.html; \
                 echo RESTORED'
}
trap restore EXIT

echo "--- checking live connections ---"
CONNS=$(ssh $KRIA 'ss -tn | grep :9099 | wc -l')
echo "active conns: $CONNS"
[ "$CONNS" -gt 0 ] && { echo "users online — aborting"; exit 1; }

echo "--- rotation off + banner up ---"
ssh precision 'systemctl --user stop kevin-rotation; \
  cd ~/kev-gpt/webchat/demo && cp client.html client.html.prebanner && \
  sed -i "s|</style></head><body>|</style></head><body><div style=\"background:#7a3b00;color:#ffe9c7;text-align:center;padding:8px 12px;font:14px monospace\">kevin borrow FPGA for quick hardware experiment — chat back in ~15 min. few word do trick, but zero word right now.</div>|" client.html && \
  grep -c "borrow FPGA" client.html'

echo "--- ship diag bitstream + NC=1 pack ---"
scp -q $BIT $KRIA:~/kevbit/mamba_pipe_first.bin || exit 1
scp -q /home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/board/pl_mamba_pipe.py \
       /home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/board/pl_mamba_diag.py \
       /home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/board/pl_axi_probe.py \
       /home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/board/pl_tok_probe.py \
       /home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/board/pl_emb_probe.py \
       $KRIA:~/kevmem/ || exit 1   # tables already staged (T=16 NC=3 pack)

echo "--- borrow: kill daemon by socket pid, flash diag ---"
BORROWED=1
ssh $KRIA 'PID=$(sudo -n ss -tlnp | grep :9099 | grep -oP "pid=\K[0-9]+" | head -1); \
  [ -n "$PID" ] && sudo -n kill $PID; sleep 2; \
  sudo -n fpgautil -b ~/kevbit/mamba_pipe_first.bin'

echo "--- DIAGNOSTIC RUN ---"
ssh $KRIA 'cd ~/kevmem && sudo -n python3 - <<PY 2>&1
import ctypes, mmap, os, struct, time
BASE=0xA0000000
fd=os.open("/dev/mem", os.O_RDWR|os.O_SYNC)
m=mmap.mmap(fd,0x1000,offset=BASE)
w=(ctypes.c_uint32*(0x1000//4)).from_buffer(m)
def wr(o,v): w[o>>2]=v&0xFFFFFFFF
def rd(o):   return int(w[o>>2])
def s32(v):  return v-(1<<32) if v>=1<<31 else v
for p in ("/sys/devices/platform/fclk0/set_rate","/sys/class/clk/fclk0/set_rate"):
    if os.path.exists(p): open(p,"w").write("80000000"); break
assert rd(0x34)==0x4D504950, "idcode"
def rd16(p): return [int(l,16) for l in open(p) if l.strip()]
wr(0x00,2)
while not (rd(0x04)>>2)&1: pass
d="/home/ubuntu/kevmem"
def load(sel,ws):
    wr(0x10,sel); wr(0x14,0)
    for v in ws: wr(0x18,int(v))
load(1,rd16(d+"/ms_t1.mem")); load(0,rd16(d+"/ms_t0.mem"))
for s_ in range(1,15): load(s_,rd16(d+"/ms_t%d.mem"%s_))
load(15,rd16(d+"/ms_t15.mem"))
import numpy as np
ref=np.load(d+"/mp_ref.npz"); toks=ref["toks"]
for st in range(3):
    for i in range(16):
        wr(0x2C,st*16+i); wr(0x30,int(toks[st][i])&0x3FF)
wr(0x20,0); wr(0x00,1)
while not rd(0x04)&1: pass
def dbg(sel,addr=0):
    wr(0x20,sel); wr(0x24,addr); rd(0x28); return s32(rd(0x28))
print("FIRST-NORM ISOLATION (layer 0 only):")
print("  first_ny  (its INPUT) : silicon %d  rtl 637" % dbg(10,1<<17))
print("  first_nout(its OUTPUT): silicon %d  rtl -56779" % dbg(11,1<<17))
print("run-wide:")
print("emb  (embedding)          : silicon %d  rtl 21248056" % dbg(4))
print("xw   (residual WRITE data): silicon %d  rtl 6078232" % dbg(7))
print("xbuf0(bank 0 contents)    : silicon %d  rtl 3339955" % dbg(6,1<<17))
print("ny   (rmsnorm INPUT)      : silicon %d  rtl 4347066" % dbg(0))
print("ng   (rmsnorm GAIN)       : silicon %d  rtl 131380880" % dbg(2,1<<17))
print("nout (rmsnorm OUTPUT)     : silicon %d  rtl 18025895" % dbg(5))
print("zx   (in_proj dequant)    : silicon %d  rtl -18018799" % dbg(14))
print("xn   (conv output)        : silicon %d  rtl 21999244" % dbg(15))
print("q8   (gemv activations)   : silicon %d  rtl 10548" % dbg(3))
PY' 2>&1 | tee /tmp/kevbuild/diag_run.log
echo "--- diag run complete (log: /tmp/kevbuild/diag_run.log) ---"
