# -*- coding:UTF-8 -*-
from src.trainer import Trainer
from src.arguments import get_args
import logging
import os
from datetime import datetime

def main():
    
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s | %(message)s', 
                        datefmt='%d-%b-%y %H:%M:%S', 
                        handlers=[
                            logging.StreamHandler()
                          ])
    logging.info(f"args: \n{args}")
    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()