#!/bin/bash
# Re-run the static/LRU configs and compare against the recorded baselines.
MODEL=/data/models/unsloth/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
BIN=/data/o35/llama.cpp/build/bin/llama-cli
MASTER=/data/o35/o35/masters
PROMPT="Q: What is the capital of France? A:"
clean() { grep -v "t/s\|Loading model\|Exiting\|^build      :\|^>$\|^$" "$1" > "$2"; }

run() {
    name=$1; static=$2; shift 2
    if [ -f /tmp/$name.done ]; then echo "=== $name skipped ==="; return; fi
    extra=""
    if [ "$static" = "S" ]; then extra="--moe-cache-static"; fi
    if [ "$static" = "C" ]; then extra="--moe-cache-static --moe-cache-cpu-fallback"; fi
    timeout 1200 $BIN -v -m $MODEL -cmoe -ngl 99 --moe-cache-mib 1024 \
        --moe-profile /tmp/prof2.csv $extra -c 2048 --temp 0 -s 42 \
        -p "$PROMPT" --no-display-prompt "$@" > /tmp/$name.txt 2> /tmp/$name.log
    echo "=== $name exit=$? ==="
    grep -Ei "ABORT|CUDA error|hit rate" /tmp/$name.log | tail -2
    touch /tmp/$name.done
}

check() {
    name=$1; ref=$2
    clean $MASTER/$ref.txt /tmp/$name.clean
    clean /tmp/$name.txt /tmp/$name.got
    n=$(diff /tmp/$name.clean /tmp/$name.got | wc -l)
    echo "$name vs $ref: diff lines = $n"
}

run clean8 S -n 8
check clean8 base8

run clean128 S -n 128
check clean128 base128

run clean512 S -n 512
check clean512 base512

run lru8 L -n 8
check lru8 base8

run lru128 L -n 128
check lru128 lru_master

run pin128 S --moe-cache-pin-mib 1990 -n 128
check pin128 clean128

run cpufb C -n 128
check cpufb clean128

# Prompt B regression. The masters live in $MASTER (stable), not in /tmp, because runB
# overwrites /tmp/$name.txt on every run. Each mode is checked against its own master
# (diff 0) and the cross-mode divergences are checked as recorded constants:
# cleanB vs cpufbB = 121 diff lines (common prefix 1514 chars), cpufbB vs allcpuB = 130
# (prefix 795), cleanB vs allcpuB = 137 (prefix 795). These are content diffs measured by
# tools/cmp.py after stripping the noise lines; the earlier figures 8 and 28 were raw-text
# diffs polluted by the `build      :` line, the spinner and the t/s line.
PROMPTB="Q: Explain how a diesel engine works. A:"

runB() {
    name=$1; shift
    if [ -f /tmp/$name.done ]; then echo "=== $name skipped ==="; return; fi
    timeout 1200 $BIN -v -m $MODEL -cmoe -ngl 99 --moe-cache-mib 1024 \
        --moe-profile /tmp/prof2.csv "$@" -c 2048 --temp 0 -s 42 \
        -p "$PROMPTB" --no-display-prompt > /tmp/$name.txt 2> /tmp/$name.log
    echo "=== $name exit=$? ==="
    grep -Ei "ABORT|CUDA error|hit rate" /tmp/$name.log | tail -2
    touch /tmp/$name.done
}

checkB() {
    name=$1; ref=$2; expect=$3
    if [ ! -f "$ref" ]; then echo "$name vs $ref: MASTER MISSING"; return; fi
    d=$(python3 /data/o35/o35/tools/cmp.py "$ref" "/tmp/$name.txt" | grep -o "diff=[0-9]*" | cut -d= -f2)
    if [ "$d" = "$expect" ]; then
        echo "$name vs $ref: content diff = $d OK"
    else
        echo "$name vs $ref: content diff = $d EXPECTED $expect  ** REGRESSION FAIL **"
    fi
}

runB cleanB --moe-cache-static
checkB cleanB $MASTER/cleanB.txt 0

runB cpufbB --moe-cache-static --moe-cache-cpu-fallback
checkB cpufbB $MASTER/cpufbB.txt 0
checkB cpufbB $MASTER/cleanB.txt 121
checkB cpufbB $MASTER/allcpuB.txt 130

runB allcpuB --moe-cache-mib 32 --moe-cache-static --moe-cache-cpu-fallback
checkB allcpuB $MASTER/allcpuB.txt 0
checkB allcpuB $MASTER/cleanB.txt 137

echo "VERIFY_DONE"
