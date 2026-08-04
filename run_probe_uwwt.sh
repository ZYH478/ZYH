cd /root/autodl-tmp/LightYOLO-master/UWWT-Dataset 2>/dev/null || { echo "NO UWWT-Dataset dir"; exit 0; }
echo "=== top ==="
ls -la
echo "=== yaml ==="
find . -maxdepth 2 -name '*.yaml' -exec sh -c 'echo "--- $1 ---"; cat "$1"' _ {} \;
echo "=== image counts ==="
for s in train val test; do
  n=$(find . -path "*images/$s/*" -type f 2>/dev/null | wc -l)
  echo "$s images: $n"
done
echo "=== label counts ==="
for s in train val test; do
  n=$(find . -path "*labels/$s/*" -type f 2>/dev/null | wc -l)
  echo "$s labels: $n"
done
echo "DONE"
