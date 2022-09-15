# -*- coding:UTF-8 -*-
from src.trainer import Trainer
from src.arguments import get_args
import logging
import os
from datetime import datetime

def main():
    
    args = get_args()
    if not os.path.exists("logs"):
        os.mkdir("logs")
    timestamp = datetime.today().strftime('%Y-%m-%d-%H-%M')
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s | %(message)s', 
                        datefmt='%d-%b-%y %H:%M:%S', 
                        handlers=[
                            logging.FileHandler(os.path.join("logs", f"{args.task_name}-{timestamp}.log")),
                            logging.StreamHandler()
                          ])
    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()