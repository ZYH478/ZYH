cd /root/autodl-tmp/neu-det-yolo26
echo "=== combo_report.json ==="
cat runs_iter21_combo_e250/combo_report.json 2>&1 || echo "NO report"
echo ""
echo "=== status.json ==="
cat runs_iter21_combo_e250/status.json 2>&1 || echo "NO status"
echo ""
echo "=== result.json per candidate ==="
find runs_iter21_combo_e250 -name 'result.json' -exec sh -c 'echo "--- $1 ---"; cat "$1"' _ {} \; 2>/dev/null
echo ""
echo "=== best.pt list ==="
find runs_iter21_combo_e250 -name 'best.pt' 2>/dev/null
echo "=== log tail ==="
tail -n 20 runs_iter21_combo_e250/train.log 2>&1 || echo "NO log"
echo "=== procs ==="
ps -eo pid,ppid,sess,etime,cmd 2>/dev/null | grep train_iter21 | grep -v grep
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
echo "=== uptime ==="
uptime
