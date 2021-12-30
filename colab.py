import time
import datetime
import webbrowser

# 1時間毎に任意のノートブックを開く
for i in range(12):
    browse = webbrowser.get('Opera')
    browse.open('https://colab.research.google.com/drive/1ZvxDKNRRlfroFqJ209KdAuAZLlwpc4yM?hl=ja#scrollTo=ufR86RlfGN-u')
    print(i, datetime.datetime.today())
    time.sleep(60*60)