from _pyexatn import *
import argparse

def parse_args(args):
    parser = argparse.ArgumentParser(description="ExaTN Python Framework.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     fromfile_prefix_chars='@')
    parser.add_argument("-L", "--location", action='store_true',
                        help="Print the path to the ExaTN install location.", required=False)
    opts = parser.parse_args(args)
    return opts


def main(argv=None):
    opts = parse_args(sys.argv[1:])
    exatnLocation = os.path.dirname(os.path.realpath(__file__))
    if opts.location:
        print(exatnLocation)
        sys.exit(0)

if __name__ == "__main__":
    sys.exit(main())
