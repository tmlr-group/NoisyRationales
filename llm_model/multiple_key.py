import json
import atexit
import signal
from threading import Lock
import os

lock = Lock()
api_key = None
apikey_manager_name = 'apikey_manager.json'

def get_least_used_key():
    global api_key
    with lock, open(apikey_manager_name, 'r+') as file:
        data = json.load(file)
        keys = sorted(data['keys'], key=lambda x: x['usage_count'])
        least_used_key = keys[0]
        least_used_key['usage_count'] += 1
        api_key = least_used_key['key']
        file.seek(0)
        json.dump(data, file, indent=4)
        file.truncate()

    return api_key

def release_key():
    print("Releasing API key...")
    global api_key
    if api_key:
        with lock, open(apikey_manager_name, 'r+') as file:
            data = json.load(file)
            for key_info in data['keys']:
                if key_info['key'] == api_key:
                    key_info['usage_count'] -= 1
                    break
            file.seek(0)
            json.dump(data, file, indent=4)
            file.truncate()
        api_key = None

def init_key_management(key_list):
    keys = {"keys": []}
    for key_v in key_list:
        key =dict()
        key["key"] =  key_v
        key["usage_count"] = 0
        keys["keys"].append(key)
    with open(apikey_manager_name, 'w') as file:
        json.dump(keys, file, indent=4)
    
def init_api_key_handling(key_list, json_name = "apikey_manager.json"):
    global api_key
    global apikey_manager_name
    
    if api_key != None:
        return api_key
    
    def handle_exit():
        print("Exiting: Releasing API key...")
        release_key()

    def handle_keyboard_interrupt(signum, frame):
        print("Interrupted: Releasing API key and exiting...")
        release_key()
        raise KeyboardInterrupt

    apikey_manager_name = json_name
    if not os.path.exists(apikey_manager_name):
        init_key_management(key_list)

    atexit.register(handle_exit)
    signal.signal(signal.SIGINT, handle_keyboard_interrupt)
    return get_least_used_key()
    