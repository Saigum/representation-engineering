mkdir -p logs
for cfg in configs/sweep/*.yaml; do
  tag=$(basename "$cfg" .yaml)
  echo ">>> $tag"
  python unlearning_.py --config "$cfg" > "logs/${tag}.log" 2>&1
  python run_inference.py --config "$cfg" > "logs/${tag}_inference.log" 2>&1
done
