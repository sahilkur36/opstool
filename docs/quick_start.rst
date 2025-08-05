.. _quickstart:

Quick Start
=============

Introduction to opstool
------------------------

.. role:: blue
.. role:: blue-bold

**opstool** is a powerful and user-friendly library designed to simplify and enhance structural analysis workflows 
with **OpenSees** and **OpenSeesPy**. 
It provides advanced tools for preprocessing, postprocessing, and visualization, making structural 
simulations more efficient and accessible.

Quick Guide
---------------------------

.. card:: 

  .. toctree::
    :maxdepth: 1

    src/quick_start/plot_model.ipynb
    examples/post/Frame2D/test_model.ipynb


Features
---------------

.. card:: :octicon:`graph;1.5em;sd-mr-1 fill-primary` Preprocessing

   - :blue-bold:`Fiber Section Meshing`: Generate detailed fiber meshes for various geometries. An example is shown below:
     `Fiber Section Mesh <src/pre/sec_mesh.ipynb>`_
   - :blue-bold:`GMSH Integration`: Import and convert `Gmsh <https://gmsh.info/>`__ models, including geometry, mesh, and physical groups.
     An example is shown below: `Converting GMSH to OpenSeesPy <src/pre/read_gmsh.ipynb>`_
   - :blue-bold:`Unit System Management`: Ensure consistency with automatic unit conversions.
     An example is shown below: `Automatic Unit Conversion <src/pre/unit_system.rst>`_
   - :blue-bold:`Quickly apply gravity loads and obtain mass and stiffness matrices`:
     An example is shown below: `Model data retrieval <src/pre/model_data.ipynb>`_
   - :blue-bold:`Load Transformation`: Simplify the application of loads.
     An example is shown below: `Load Transformation <src/pre/loads.ipynb>`_
   - :blue-bold:`Tcl script to OpenSeesPy`: Convert Tcl scripts to OpenSeesPy scripts.
     An example is shown below: `Tcl to OpenSeesPy <src/pre/tcl2py.rst>`_


.. card:: :octicon:`database;1.5em;sd-mr-1 fill-primary` Postprocessing
   
  Easy retrieval and interpretation of analysis results using `xarray <https://docs.xarray.dev/en/stable/index.html#>`__.

  All examples are shown below: `Postprocessing with xarray <src/post/index.ipynb>`_

  The following is a non-exhaustive list of response types supported by this library:

  .. tab-set::
    :sync-group: category

    .. tab-item:: Nodal
        :sync: key1

        ✅ "disp"
        ✅ "vel"
        ✅ "accel"
        ✅ "reaction"
        ✅ "reactionIncInertia"
        ✅ "rayleighForces"
        ✅ "pressure"

    .. tab-item:: Frame
        :sync: key2

        ✅ "localForces"
        ✅ "basicForces"
        ✅ "basicDeformations"
        ✅ "plasticDeformation"
        ✅ "sectionForces"
        ✅ "sectionDeformations"
        ✅ "sectionLocs"
    
    .. tab-item:: Truss
        :sync: key3

        ✅ "axialForce" ✅ "axialDefo" ✅ "Stress" ✅ "Strain"
    
    .. tab-item:: Link
        :sync: key4

        ✅ "basicDeformation" ✅ "basicForce"
    
    .. tab-item:: Fiber Section
        :sync: key5

        ✅ "Stresses" ✅ "Strains" ✅ "secForce" ✅ "secDefo"
    
    .. tab-item:: Shell
        :sync: key6

        ✅ "sectionForces"
        ✅ "sectionDeformations"
        ✅ "Stresses"
        ✅ "Strains"
        ✅ "sectionForcesAtNodes"
        ✅ "sectionDeformationsAtNodes"
        ✅ "StressesAtNodes"
        ✅ "StrainsAtNodes"
    
    .. tab-item:: Plane
        :sync: key7

        ✅ "Stresses"
        ✅ "Strains"
        ✅ "StressesAtNodes"
        ✅ "StressAtNodesErr"
        ✅ "StrainsAtNodes"
        ✅ "StrainsAtNodesErr"
        ✅ "StressMeasures"
        ✅ "StrainMeasures"
        ✅ "StressMeasuresAtNodes"
        ✅ "StrainMeasuresAtNodes"
    
    .. tab-item:: Solid
        :sync: key8

        ✅ "Stresses"
        ✅ "Strains"
        ✅ "StressesAtNodes"
        ✅ "StressAtNodesErr"
        ✅ "StrainsAtNodes"
        ✅ "StrainsAtNodesErr"
        ✅ "StressMeasures"
        ✅ "StrainMeasures"
        ✅ "StressMeasuresAtNodes"
        ✅ "StrainMeasuresAtNodes"
    
    .. tab-item:: Contact
        :sync: key9

        ✅ "globalForces" ✅ "localForces" ✅ "localDisp" ✅ "slips"
    
    .. tab-item:: Sensitivity Analysis
        :sync: key10

        ✅ "disp"
        ✅ "vel"
        ✅ "accel"
        ✅ "pressure"
        ✅ "lambdas"

.. card:: :octicon:`eye;1.5em;sd-mr-1 fill-primary` Visualization
   
   - Powered by `PyVista <https://docs.pyvista.org/>`__ (VTK-based) and `Plotly <https://plotly.com/python/>`__ (web-based).
   - Nearly identical APIs for flexible visualization of model geometry, modal analysis, and simulation results.
   - Supports most common OpenSees elements.
   - An example is shown below: `Visualization <src/vis/index.ipynb>`_

.. card:: :octicon:`paper-airplane;1.5em;sd-mr-1 fill-primary` Intelligent Analysis

   - Features like :blue-bold:`automatic step size adjustment` and :blue-bold:`algorithm switching` to optimize simulation workflows.
     An example is shown below: `Intelligent Analysis <src/analysis/smart_analysis.rst>`_
   - :blue-bold:`Moment-Curvature Analysis`: Generate moment-curvature curves for various sections.
     An example is shown below: `Moment-Curvature Analysis <src/analysis/mc_analysis.ipynb>`_
   - :blue-bold:`Linear Buckling Analysis`: Perform linear buckling analysis for stability assessment.
     An example is shown below: `Linear Buckling Analysis <src/analysis/buckling_analysis_linear.ipynb>`_

