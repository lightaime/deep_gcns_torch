import subprocess
import os
import sys


def find_last_line(path, exp):
    ans = []
    if not os.path.exists(path):
        raise Exception('Path does not exist')
    for r, d, f in os.walk(path):
        for file in f:
            fullfile = os.path.join(r,file)
            line = subprocess.check_output(['tail', '-10', fullfile])
            match = line.find(bytearray(exp, 'utf-8'))
            if (match != -1):
                ans.append((fullfile, line.decode()))
    return ans


def main(argv):
    path = argv[0]
    exp = argv[1]
    matches = find_last_line(path, exp)
    if len(matches) == 0:
        print("Couldn't find any match")
    else:
        for match in matches:
            print('{} in file {}'.format(match[1], match[0]))

if __name__ == "__main__":
    main(sys.argv[1:])


