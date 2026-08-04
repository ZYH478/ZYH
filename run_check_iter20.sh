cd /root/autodl-tmp/neu-det-yolo26
echo "=== result.json ==="
cat runs_gsdown_dwr_e250/result.json 2>&1 || echo "NO result.json"
echo ""
echo "=== find best.pt ==="
find runs_gsdown_dwr_e250 -name 'best.pt' 2>/dev/null
echo "=== log tail ==="
tail -n 25 runs_gsdown_dwr_e250/gsdown_dwr_e250.log 2>&1 || echo "NO log"
echo "=== procs ==="
ps -eo pid,ppid,sess,etime,cmd 2>/dev/null | grep -E 'train_gsdown_dwr|train_iter21' | grep -v grep
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
echo "=== uptime ==="
uptime
