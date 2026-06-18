source venv1/bin/activate
cd src

models=(
"trainning.squeezenet_train"
)

seeds=(42 43 44 45 46)

for model in "${models[@]}"; do
  for seed in "${seeds[@]}"; do
    python -m "$model" --seed "$seed"
  done
done