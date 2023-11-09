# FAILED

import requests

url = "http://www.utic.go.kr/view/map/openDataCctvStream.jsp?key=EJZFV8qgGajzoSgH1ix9D9VOyzQ1wW9grum5DFKKiShlgPCprL3rmGUxPeMHgnSl&cctvid=L280044&cctvName=%25EC%259E%25A5%25EC%258A%25B9%25ED%258F%25AC%25EB%258F%2599%2520%25EB%2591%2590%25EB%25AA%25A8%25EB%25A1%259C%25ED%2584%25B0%25EB%25A6%25AC&kind=t&cctvip=null&cctvch=7&id=637bef0325205&cctvpasswd=07&cctvport=null"
response = requests.get(url=url)
html = response.text



from bs4 import BeautifulSoup

soup = BeautifulSoup(html, "html.parser")

iframe_src = soup.find("iframe")["src"]



import cv2

stream_url = iframe_src

cap = cv2.VideoCapture(filename=stream_url)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow(winname='real-time CCTV video', mat=frame)

    key = cv2.waitKey(delay=1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()