import tarfile

print("Extracting dataset")
tar = tarfile.open("./RML2016.10b.tar.bz2")
tar.extractall()
tar.close()
print("Done")

