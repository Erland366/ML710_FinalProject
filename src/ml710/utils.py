def create_logger(pgm=None, name="app"):
    import logging
    import sys
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    logger.handlers = []
    
    handler = logging.StreamHandler(sys.stdout)
    
    if pgm is not None:
        try:
            prefix = f"[{pgm.process_group_manager}]"
        except:
            prefix = f"[{name}]"
    else:
        prefix = f"[{name}]"
    
    formatter = logging.Formatter(f'{prefix} %(message)s')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger