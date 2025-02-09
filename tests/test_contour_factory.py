from pathlib import Path
from cfplot4.cfcontour import ContourFactory
import cf
from matplotlib import pyplot as plt

HERE = here = Path(__file__).parent
DATA_DIR = HERE/'test_data'

def test_two_plots():

    files = DATA_DIR.glob('*.nc')
    files = list(files)
    fig, axes = plt.subplots(nrows=1, ncols=len(files))
    axen = axes.flatten()

    for i,f in enumerate(files):

        fld = cf.read(f)[0]
        con = ContourFactory.create(fld,ax=axen[i])

    plt.savefig('test_two_plots.png')
