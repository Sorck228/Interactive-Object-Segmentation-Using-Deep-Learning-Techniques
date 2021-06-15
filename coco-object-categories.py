cat_2017 = '../../Github_data/coco/annotations/instances_val2017.json'

import sys, getopt
import json

def main(argv):
    json_file = None
    file1 = open("coco_cat.txt", "a")#append mode
    try:
        opts, args = getopt.getopt(argv,"hy:")
    except getopt.GetoptError:
        print('coco_categories.py -y <year>')
        sys.exit(2)
    json_file = cat_2017
    if json_file is not None:
        with open(json_file,'r') as COCO:
            js = json.loads(COCO.read())
            print(json.dumps(js['categories']))
            file1.write(json.dumps(js['categories']))


    file1.close()
if __name__ == "__main__":
    main(sys.argv[1:])

