import os
import subprocess

def cmd_exists(cmd):
    return (subprocess.call("type " + cmd, shell=True, 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0)

if __name__ == "__main__":
	print("convert_imageset exist : " + str(cmd_exists("convert_imageset")))
	print("compute_image_mean exist : " + str(cmd_exists("compute_image_mean")))
