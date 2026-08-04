cd /root/autodl-tmp/neu-det-yolo26
echo "=== base ==="
ls -1 runs_module_sweep_e250/y26n_base_e250/weights/best.pt 2>/dev/null || echo MISS
echo "=== winner spd_p3_dysample ==="
find . -path '*spd_p3_dysample*/weights/best.pt' 2>/dev/null | grep -vE 'iter21'
echo "=== gsdown vovgscsp ==="
find . -path '*vovgscsp_gsdown*/weights/best.pt' 2>/dev/null | grep -vE 'iter21|dwr|aux'
echo "=== dwr_deep (iter19) ==="
find . -path '*dwr_deep*/weights/best.pt' 2>/dev/null
find . -path '*iter19*/weights/best.pt' 2>/dev/null
echo "=== all iter19 runs dirs ==="
ls -d runs_iter19* 2>/dev/null
find runs_iter19* -name best.pt 2>/dev/null
echo "DONE"
