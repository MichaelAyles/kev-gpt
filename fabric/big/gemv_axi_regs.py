"""Register map of `rtl/gemv_axi.v` — the R2a board wrapper around `gemv_stream`.

Single source of truth for the RTL (the testbench checks the RTL against it), the sim gate
(`run_gemv_axi.py`) and the board driver (`board/pl_gemv.py`). No dependencies, so the Kria can
import it.

The wrapper = AXI-Lite slave (this map) + NPORTS AXI4 128-bit read masters, each feeding a FIFO
into one `gemv_stream` port, + a result capture RAM + counters. One run = one matrix:

    load the activation (ACT_* registers, one 128-group at a time)
    write ROWS / GROUPS / PAD / BURSTS / CHUNK / STEP / MAXOUT / BASE / PORT_OFF[k] / RES_START
    write CTRL.GO, poll STATUS.DONE, read the result window + counters

Port k issues BURSTS INCR bursts of CHUNK beats, burst n at
    BASE + PORT_OFF[k] + n * STEP
(single ARID, so data returns in order). With the packer's layout (`pack_weights.py` §4,
align = "stripe") that is BASE = image base + (T0 // CHUNK) * SLOTS*CHUNK*16,
PORT_OFF[k] = PORT_SLOTS[k]*CHUNK*16, STEP = SLOTS*CHUNK*16, BURSTS = ceil(tuples / CHUNK),
PAD = BURSTS*CHUNK - R*TPR.
"""

BASE_ADDR = 0xA000_0000          # where the block design maps the slave
ADDR_BITS = 17                   # 128 KB: registers in [0, 0x10000), result window above
IDCODE_VALUE = 0x47454D56        # "GEMV"
VERSION = 1

# ---- control / status
CTRL = 0x000          # W  bit0 GO (pulse; ignored while busy)   bit1 SOFT_RST (write 1, then 0)
STATUS = 0x004        # R  bit0 busy  bit1 done (sticky until GO / reset)  bit2 axi_err (sticky)
                      #    bit3 fifo_overflow (sticky; must never happen)
IDCODE = 0x008        # R  IDCODE_VALUE
PARAMS = 0x00C        # R  [3:0] NPORTS  [15:8] GMAX  [23:16] log2(RES_WORDS)  [31:24] VERSION

# ---- run configuration (static while busy)
ROWS = 0x010          # RW output rows R of the matrix
GROUPS = 0x014        # RW 128-groups per row G (1..GMAX)
PAD = 0x018           # RW pad tuples to discard after the matrix
BURSTS = 0x01C        # RW bursts per port for this run
CHUNK = 0x020         # RW beats per burst, power of two in 1..256
STEP = 0x024          # RW bytes between successive bursts of one port
MAXOUT = 0x028        # RW max bursts in flight per port (further limited by FIFO credits)
RES_START = 0x02C     # RW first result row captured into the result window
BASE_LO = 0x030       # RW matrix base byte address, CHUNK*16 aligned
BASE_HI = 0x034       # RW bits [48:32]
PORT_OFF = 0x040      # RW PORT_OFF + 4*k, k = 0..7: byte offset of port k's first burst from BASE

# ---- activation load (only while !busy)
ACT_GROUP = 0x060     # RW group index
ACT_EXP = 0x064       # RW signed exponent of that group (low EXW bits used)
ACT_COMMIT = 0x068    # W  any write commits {ACT_MANT[0..39], ACT_EXP} to group ACT_GROUP
ACT_MANT = 0x100      # RW ACT_MANT + 4*w, w = 0..39: bits [32w+31 : 32w] of the 1280-bit word;
ACT_MANT_WORDS = 40   #    element j of the group = bits [10j+9 : 10j], two's complement

# ---- counters (cleared by GO)
CYC_LO = 0x200        # R  cycles from GO to the last result row
CYC_HI = 0x204
CYC_STREAM_LO = 0x208  # R  cycles from the first AR handshake to the last R beat (any port)
CYC_STREAM_HI = 0x20C
ROWS_DONE = 0x210     # R  result rows produced
SUM_LO = 0x214        # R  64-bit sum of all result words (each taken as unsigned 32-bit)
SUM_HI = 0x218
XOR = 0x21C           # R  xor of all result words
TUPLES = 0x220        # R  tuples popped by the engine
STALL_IN = 0x224      # R  busy cycles with no tuple popped because some port FIFO was empty
STALL_OUT = 0x228     # R  busy cycles with a tuple available but no output credit
ERRCNT = 0x22C        # R  R beats with RRESP != OKAY

# ---- per port k (0..7): PORT_BASE + PORT_STRIDE*k + ...
PORT_BASE = 0x300
PORT_STRIDE = 0x20
P_BEATS = 0x00        # R  beats received
P_EMPTY = 0x04        # R  busy cycles this port's FIFO was the (or an) empty one
P_FIFO_MAX = 0x08     # R  maximum FIFO fill seen
P_XSUM = 0x10         # R  P_XSUM + 4*i, i = 0..3: 128-bit xor of every beat received

# ---- result window
RES_BASE = 0x10000    # R  RES_BASE + 4*i = result row RES_START + i (raw fp32 bits)
RES_WORDS = 16384     #    rows captured per run


def port_reg(k: int, off: int) -> int:
    return PORT_BASE + PORT_STRIDE * k + off
