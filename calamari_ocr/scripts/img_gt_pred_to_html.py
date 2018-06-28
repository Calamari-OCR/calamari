import argparse
import webbrowser

from calamari_ocr.utils import glob_all, split_all_ext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True,
                        help="The image files to predict with its gt and pred")
    parser.add_argument("--html_output", type=str, required=True,
                        help="Where to write the html file")
    parser.add_argument("--open", action="store_true",
                        help="Automatically open the file")

    args = parser.parse_args()
    img_files = sorted(glob_all(args.files))
    gt_files = [split_all_ext(f)[0] + ".gt.txt" for f in img_files]
    pred_files = [split_all_ext(f)[0] + ".pred.txt" for f in img_files]

    with open(args.html_output, 'w') as html:
        html.write("""
                   <!DOCTYPE html>
                   <html lang="en">
                   <head>
                       <meta charset="utf-8"/>
                   </head>
                   <body>
                   <ul>""")

        for img, gt, pred in zip(img_files, gt_files, pred_files):
            html.write("<li><p><img src=\"file://{}\"></p><p>{}</p><p>{}</p>\n".format(
                img.replace('\\', '/').replace('/', '\\\\'), open(gt).read(), open(pred).read()
            ))

        html.write("</ul></body></html>")

    if args.open:
        webbrowser.open(args.html_output)


if __name__ == "__main__":
    main()
