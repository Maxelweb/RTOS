import os
import numpy as np
from scipy.interpolate import splrep, splev
# ---- EXTRA FOR DOCKER ----
import matplotlib
matplotlib.use('Agg') # avoid using the backend instaed of Xwindows
# ---- ----
import matplotlib.pyplot as plt
from toolbox.datafile import *

from matplotlib import rc

RESULTS_DATA_COMPLETE_DIR = "./data_complete/"
PDFS_COMPLETE_DIR = "./pdfs_dir/"
LATEX_GRAPHIC_TEMPLATE = "./pdf_latex_template.tex"
LATEX_PDF_RELATIVE_OUTPUT_DIR = "INSERT_DIR_TO_BE_PLACED_IN_LATED_TEMPLATE"
LATEX_OUTPUT_FILE = "INSERT_OUTPUT_DIR_FOR_TEX_FILE_OF_IMAGES"

def get_complete_experiment_files():

    for filename in os.listdir(RESULTS_DATA_COMPLETE_DIR):
        if filename.endswith(".csv"):
            yield filename

def produce_plot(source_dir, destination_dir, file_name, make_latex_caption=False):

    f = DataFile(source_dir + file_name)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=80)

    x_axis = [z[0] for z in f.data]
    y_gipp = []
    y_omip = []
    y_rnlp = []

    for row in f.data:
        # first element of row is critical section length
        # correct by subtracting 1
        samples = len(row)-1
        gipp_passed = 0
        omip_passed = 0
        rnlp_passed = 0

        for i in xrange(1, len(row)):
            passes = [int(x) for x in row[i].split('-')]
            if passes[0] == 1:
                gipp_passed += 1
            if passes[1] == 1:
                omip_passed += 1
            if passes[2] == 1:
                rnlp_passed += 1

        y_gipp.append(float(gipp_passed) / float(samples))
        y_omip.append(float(omip_passed) / float(samples))
        y_rnlp.append(float(rnlp_passed) / float(samples))

    ax.plot(x_axis, y_gipp, 'bD-', markevery=(5, 15), label="GIPP")
    ax.plot(x_axis, y_omip, 'gs-', markevery=(10, 15), label="OMIP")
    ax.plot(x_axis, y_rnlp, 'ro-', markevery=(15, 15), label="CA-RNLP")

    ax.axis([5, 1000, 0, 1.0])
    _legend = ax.legend(loc='upper right')

    U = (float(f.params["lo"]) * float(f.params["m"]))

    title = r'$\mathregular{m=%d}$  ' % int(f.params["m"])
    title += r'$\mathregular{U=%.2f}$  ' % (U)
    title += r'$\mathregular{n=%d}$  ' % int(f.params["n"])
    title += r'$\mathregular{n^{ls}=%d}$  ' % int(f.params["ls"])
    title += r'$\mathregular{q=%d}$  ' % int(f.params["res-nls"])
    title += r'$\mathregular{g^{size}=%d}$ ' % int(f.params["gs-nls"])

    # groups of size 2 or less don't have both a wide and deep variant (as its not possible)
    if int(f.params["gs-nls"]) > 2:
        title += r'$\mathregular{g^{type}=%s}$ ' % ('W' if int(f.params["gt-nls"]) == 0 else 'D')
    
    title += r'$\mathregular{N^{max}=%d}$  ' % int(f.params["acc-nls"])
    
    ax.set(
            title=title,
            ylabel="fraction of schedulable task sets",
            xlabel=r"maximum critical section length (in $\mathregular{\mu}$s) of non-latency-sensitive tasks"
    )

    asymm = 0
    if "asym" in f.params:
        asymm = int(f.params["asym"])

    new_filename = "exp-"
    new_filename += "m%02d-" % int(f.params["m"])
    new_filename += "U%s-" % "{:05.2f}".format(U).replace(".", "")
    new_filename += "n%02d-" % int(f.params["n"])
    new_filename += "ls%02d-" % int(f.params["ls"])
    new_filename += "resnls%02d-" % int(f.params["res-nls"])
    new_filename += "resls%02d-" % int(f.params["res-ls"])
    new_filename += "accnls%02d-" % int(f.params["acc-nls"])
    new_filename += "accls%02d-" % int(f.params["acc-ls"])
    new_filename += "gsnls%02d-" % int(f.params["gs-nls"])
    new_filename += "gtnls%02d-" % int(f.params["gt-nls"])
    
    new_filename += 'tp%s-' % "{:02.1f}".format(float(f.params["tp"])).replace(".", "")
    new_filename += "asym%02d" % asymm

    new_filename += ".pdf"

    caption = ""
    if make_latex_caption:
        caption += r"Schedulability under the GIPP, OMIP, and CA-RNLP for the labelled parameters."

    plt.savefig(destination_dir + new_filename)
    plt.close(fig)

    return (new_filename, caption)


if __name__ == "__main__":

    plt.rcParams.update({'font.size': 13})

    make_latex = True
    data_files = []
    plot_files = []

    data_files = []
    
    for x in get_complete_experiment_files():
        data_files.append(x)

    for x in sorted(data_files):

        plot_file = produce_plot(RESULTS_DATA_COMPLETE_DIR, PDFS_COMPLETE_DIR, x, make_latex)

        if make_latex:
            plot_files.append(plot_file)
    
    if make_latex:

        print "\n"
        current_on_row = 0
        figures_per_row = 2

        latex_file = open(LATEX_OUTPUT_FILE, "w")
        t = open(LATEX_GRAPHIC_TEMPLATE, 'r')
        template = t.read()
        t.close()

        for plot_file, caption in sorted(plot_files, key=lambda x: x[0]):

            print x
            data = ""

            if current_on_row == 0:
                data += r"\begin{figure}[H]" + "\n"

            data += template.replace("__FILE_PATH__", LATEX_PDF_RELATIVE_OUTPUT_DIR + plot_file)
            data = data.replace("__CAPTION__", caption)
            
            current_on_row += 1
            if current_on_row == figures_per_row:
                print "end figure"
                data += r"\end{figure}" + "\n\n"
                current_on_row = 0
            else:
                data += r"\hfill" + "\n"

            latex_file.write(data)

        if current_on_row != 0:
            latex_file.write(r"\end{figure}" + "\n\n")
        
        latex_file.close()
        
