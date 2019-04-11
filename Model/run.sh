nohup python -u main.py > log.txt 2>&1 &
tail -f log.txt
