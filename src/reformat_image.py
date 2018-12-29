from PIL import Image
import argparse
import os

parser = argparse.ArgumentParser(description='reformat images to have the same size')
parser.add_argument('file_name', type=str, metavar='File_name',
                    help='The name of the file to be reformatted')

directory = "../data/original"
file_name = parser.parse_args().file_name
path = os.path.join(directory, file_name)


img = Image.open(path)
img = img.resize((200, 150), Image.ANTIALIAS)
img.save(path)