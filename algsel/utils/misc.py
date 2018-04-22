from time import gmtime, strftime

def get_current_time():
    return strftime("[%Y-%m-%d %H:%M:%S]", gmtime())