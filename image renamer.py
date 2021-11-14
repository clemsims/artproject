import os
os.getcwd()
collection = "./painting_frames"

print(os.listdir("./painting_frames"))


""""

print(enumerate(os.listdir(collection)))
for i, filename in enumerate(os.listdir(collection)):
    os.rename("./painting_frames/" + filename, "./painting_frames" + str(i) + ".jpg")

"""