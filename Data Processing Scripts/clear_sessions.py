import os, glob
[os.remove(f) for f in glob.glob("session_*.csv")]