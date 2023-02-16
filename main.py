import argparse
from unicodedata import name

from detector import *


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-id', '--image_directory', type=str,
                        default='test_folder',
                        help='name of the dir, images will be downloaded')

    parser.add_argument('-ujp', '--url_json_path', type=str,
                        default=None, help='Path to the json file of urls')

    parser.add_argument('-tjp', '--target_json_path', type=str,
                        default=None, help='Path to the json file of target brands')

    parser.add_argument('-p', '--precision', type=int,
                        default=1, help='If set to 1 results will be more precise but it will take more time')

    args = parser.parse_args()

    model = Detector(image_path=args.image_directory, url_json_path=args.url_json_path,
                     target_brands_path=args.target_json_path)

    if not args.precision:
        rotations = None
    else:
        rotations = [-45, -90, 45, 90]

    return_dict = model.transform(rotations=rotations)
    return return_dict


if __name__ == '__main__':
    result = main()
    print(result)
