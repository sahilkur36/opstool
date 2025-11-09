r"""
606 m Mega-tall Building
==========================

See the original ``tcl`` file of the model in `OpenSees models for a 606 m Mega-tall Building <http://www.luxinzheng.net/download/OpenSEES/Mega-tall_Building_Benchmark_OpenSees.htm>`_,
and a ``python`` file converted by :func:`opstool.pre.tcl2py` that you can download from here
https://ln5.sync.com/dl/9e3582d40/2s3k7ayf-e52anp3t-mjfa89vz-jc427xsb named ``MegatallBuilding.py``.
"""

import time

from utils.MegatallBuilding import *

import opstool as opst

# %%
start = time.time()
opst.post.save_model_data(odb_tag="megatall-building")
end = time.time()
print("Data Saving: Time elapsed: ", end - start, " s")

# %%
opst.vis.pyvista.set_plot_props(point_size=0, line_width=0.5)

start = time.time()
plotter = opst.vis.pyvista.plot_model(odb_tag="megatall-building")
plotter.show()
# plotter.show()
end = time.time()
print("Pyvista-Based: Time elapsed: ", end - start, " s")

# %%
opst.vis.plotly.set_plot_props(point_size=0, line_width=0.4)

start = time.time()
fig = opst.vis.plotly.plot_model(odb_tag="megatall-building", show_outline=False)
# fig.show(renderer="browser")
end = time.time()
print("Plotly-Based: Time elapsed: ", end - start, " s")
fig
