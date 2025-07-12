import requests

youtube_urls = ["https://www.youtube.com/watch?v=Qzl2wvW8R7A&list=RDQzl2wvW8R7A&start_radio=1", 
                "https://www.youtube.com/watch?v=3ewtjbyJFt4&list=RDMM&start_radio=1&rv=Qzl2wvW8R7A",
                "https://www.youtube.com/watch?v=bPfhUpwdN-g&list=RDMM&index=2",
                "https://www.youtube.com/watch?v=jBE_X3cMCyM&list=RDMM&index=4",
                "https://www.youtube.com/watch?v=ipCCl3jvZC0",
                "https://www.youtube.com/watch?v=GX8Hg6kWQYI",
                "https://www.youtube.com/watch?v=mzB1VGEGcSU",
                "https://www.youtube.com/watch?v=pgN-vvVVxMA",
                "https://www.youtube.com/watch?v=-jRKsiAOAA8",
                "https://www.youtube.com/watch?v=WvV5TbJc9tQ",
                "https://www.youtube.com/watch?v=h3EJICKwITw"
                   ]

for url in youtube_urls:
    response = requests.post("http://localhost:5000/convert", json={"url": url})

print("Status:", response.status_code)
print("Resposta:", response.json())
