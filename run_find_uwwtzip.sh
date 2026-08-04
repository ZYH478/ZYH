echo "=== autodl-fs ==="
ls -la /root/autodl-fs/ 2>/dev/null || echo "NO autodl-fs"
echo "=== find zip ==="
find /root/autodl-fs -maxdepth 2 -iname '*uwwt*' 2>/dev/null
find /root -maxdepth 3 -iname 'UWWT-Dataset-1500.zip' 2>/dev/null
echo "DONE"
