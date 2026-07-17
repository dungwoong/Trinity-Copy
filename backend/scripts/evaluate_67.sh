
echo "**********************[LLaMA Keyformer A]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_figure67.py --m llama --t keyformer --n A
    sleep 5
done

echo "**********************[LLaMA Keyformer B]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_figure67.py --m llama --t keyformer --n B
    sleep 5
done

echo "**********************[LLaMA Keyformer C]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_figure67.py --m llama --t keyformer --n C
    sleep 5
done

echo "**********************[LLaMA Keyformer D]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_figure67.py --m llama --t keyformer --n D
    sleep 5
done

echo "**********************[LLaMA Keyformer E]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_figure67.py --m llama --t keyformer --n E
    sleep 5
done

echo "**********************[LLaMA Keyformer F]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_figure67.py --m llama --t keyformer --n F
    sleep 5
done

echo "**********************[LLaMA Keyformer G]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_figure67.py --m llama --t keyformer --n G
    sleep 5
done

echo "**********************[LLaMA Keyformer H]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_figure67.py --m llama --t keyformer --n H
    sleep 5
done

echo "**********************[LLaMA RoCo A]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_figure67.py --m llama --t roco --n A
    sleep 5
done

echo "**********************[LLaMA RoCo B]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_figure67.py --m llama --t roco --n B
    sleep 5
done

echo "**********************[LLaMA RoCo C]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_figure67.py --m llama --t roco --n C
    sleep 5
done

echo "**********************[LLaMA RoCo D]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_figure67.py --m llama --t roco --n D
    sleep 5
done

echo "**********************[LLaMA RoCo E]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_figure67.py --m llama --t roco --n E
    sleep 5
done