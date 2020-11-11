import musdb
mus = musdb.DB(root='/Users/vishrud/Desktop/Vasanth/Technology/Mobile-ML/Spleeter_TF2.0/musdb_dataset/train/')

for track in mus:
    print(track.audio)