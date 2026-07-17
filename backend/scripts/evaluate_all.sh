#!/bin/bash

GPU="$1"
option=2

echo "Running with option = $option"

if [ "$GPU" == "5090" ]; then
    echo "Running for RTX5090..."    

    echo "**********************[LLaMA Vanilla]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m llama --t vanilla --n 946
        sleep 5
    done
    echo "**********************[LLaMA PreNorm]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m llama --t prenorm --n 579
        sleep 5
    done
    echo "**********************[LLaMA QKNorm]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m llama --t qknorm --n 2321
        sleep 5
    done
    echo "**********************[LLaMA KeyFormer]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m llama --t keyformer --n 2754
        sleep 5
    done
    echo "**********************[LLaMA RoCo]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m llama --t roco --n 1753
        sleep 5
    done
    echo "**********************[LLaMA FFN]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m llama --t ffn --n 2248
        sleep 5
    done

    echo "**********************[Falcon Vanilla]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m falcon --t vanilla --n 1562
        sleep 5
    done
    echo "**********************[Falcon PreNorm]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m falcon --t prenorm --n 579
        sleep 5
    done
    echo "**********************[Falcon QKNorm]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m falcon --t qknorm --n 3690
        sleep 5
    done
    echo "**********************[Falcon KeyFormer]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m falcon --t keyformer --n 3323
        sleep 5
    done
    echo "**********************[Falcon RoCo]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m falcon --t roco --n 1620
        sleep 5
    done
    echo "**********************[Falcon FFN]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m falcon --t ffn --n 2248
        sleep 5
    done

elif [ "$GPU" == "A100" ]; then
    echo "Running for A100..."    

    echo "**********************[LLaMA Vanilla]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m llama --t vanilla --n 1562
        sleep 5
    done
    echo "**********************[LLaMA PreNorm]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m llama --t prenorm --n 579
        sleep 5
    done
    echo "**********************[LLaMA QKNorm]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m llama --t qknorm --n 3690
        sleep 5
    done
    echo "**********************[LLaMA KeyFormer]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m llama --t keyformer --n 1837
        sleep 5
    done
    echo "**********************[LLaMA RoCo]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m llama --t roco --n 1620
        sleep 5
    done
    echo "**********************[LLaMA FFN]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m llama --t ffn --n 2248
        sleep 5
    done

    echo "**********************[Falcon Vanilla]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m falcon --t vanilla --n 1562
        sleep 5
    done
    echo "**********************[Falcon PreNorm]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m falcon --t prenorm --n 579
        sleep 5
    done
    echo "**********************[Falcon QKNorm]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m falcon --t qknorm --n 1350
        sleep 5
    done
    echo "**********************[Falcon KeyFormer]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m falcon --t keyformer --n 2682
        sleep 5
    done
    echo "**********************[Falcon RoCo]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m falcon --t roco --n 4775
        sleep 5
    done
    echo "**********************[Falcon FFN]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m falcon --t ffn --n 2248
        sleep 5
    done

elif [ "$GPU" == "H100" ]; then
    echo "Running for H100..."    

    echo "**********************[LLaMA Vanilla]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m llama --t vanilla --n 1562
        sleep 5
    done
    echo "**********************[LLaMA PreNorm]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m llama --t prenorm --n 579
        sleep 5
    done
    echo "**********************[LLaMA QKNorm]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m llama --t qknorm --n 3690
        sleep 5
    done
    echo "**********************[LLaMA KeyFormer]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m llama --t keyformer --n 2430
        sleep 5
    done
    echo "**********************[LLaMA RoCo]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m llama --t roco --n 1620
        sleep 5
    done
    echo "**********************[LLaMA FFN]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m llama --t ffn --n 2248
        sleep 5
    done

    echo "**********************[Falcon Vanilla]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m falcon --t vanilla --n 1562
        sleep 5
    done
    echo "**********************[Falcon PreNorm]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m falcon --t prenorm --n 579
        sleep 5
    done
    echo "**********************[Falcon QKNorm]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m falcon --t qknorm --n 1350
        sleep 5
    done
    echo "**********************[Falcon KeyFormer]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m falcon --t keyformer --n 1799
        sleep 5
    done
    echo "**********************[Falcon RoCo]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m falcon --t roco --n 4775
        sleep 5
    done
    echo "**********************[Falcon FFN]****************************"
    for i in {1..3}; do
        echo "<Trial $i>"
        python run_eval.py --o "$option" --m falcon --t ffn --n 2248
        sleep 5
    done
fi