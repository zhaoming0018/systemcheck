#!/bin/bash

# 测试是否存在第一个参数
if [ ! -n "$1" ]; then
    echo "必须指定攻击类型"
    exit
fi

attack=$1
echo "attack is: $attack"

DIR="train-log/$attack";
if [ ! -d "$DIR" ]; then
    # If it doesn't create it
    mkdir -p $DIR
fi

case $attack in
    'cap' )
    for mal_p in 0.1 0.2 0.4 0.8 1.6
    do
        printf "run command: \n\tpython train.py --attack=$attack --mal_p=$mal_p 2>&1 | tee $DIR/$attack-train-$mal_p.log\n"
    	python train.py --attack=$attack --mal_p=$mal_p 2>&1 | tee $DIR/$attack-train-$mal_p.log
    done
    ;;
    'cor' )
    for corr in 1.0 2.0 4.0 8.0 16.0
    do
        printf "run command: \n\tpython train.py --attack=$attack --corr=$corr 2>&1 | tee $DIR/$attack-train-$corr.log\n"
    	python train.py --attack=$attack --corr=$corr 2>&1 | tee $DIR/$attack-train-$corr.log
    done
    ;;
    'sgn' )
    for corr in 8.0 16.0 32.0 64.0
    do
        printf "run command: \n\tpython train.py --attack=$attack --corr=$corr 2>&1 | tee $DIR/$attack-train-$corr.log\n"
    	python train.py --attack=$attack --corr=$corr 2>&1 | tee $DIR/$attack-train-$corr.log
    done
    ;;
esac

