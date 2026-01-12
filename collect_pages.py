import requests
from typing import *
import time

prev_url = {
"https://s2-mha1-nlams.bzcdn.net/scomic/wodetudidushidafanpai-yuetian/0/464-jrwf"
}

img_url :str = "https://s1.bzcdn.net/scomic/yuqiangzeqiangwodexiuweiwushangxian-huolongguomanhua/0/110-c038"

start:int = 3
end:int = 51

def main(url : str , start:int , end:int):
    for req_no in range(start , end):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
                }
            response = requests.get(
                f"{url}/{req_no}.jpg" , 
                headers=headers
            )
            print(
                f"getting response from req {req_no} : {response}"
            )
            if response.status_code == 200:
                write_img_bytes(response.content , req_no)
            else:
                print("Getting Status Code : " , response.status_code)
                break
            time.sleep(5)
        except Exception as e:
            print("Getting error :" , e)
            return

def write_img_bytes(data:bytes , img_no:int):
    with open(f"t_01/img_{str(img_no).zfill(2)}.png" , "+wb") as f:
        f.write(data)
    
    print(f"image {img_no} write is completed...")

print(f"ESTIMATED TIME : {((end - start) * 6) / 60} MIN")
main(url=img_url , start=start , end=end)