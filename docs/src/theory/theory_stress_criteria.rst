.. _theory_stress_criteria:

=============================
Stress and Strength Criteria
=============================

This section provides a comprehensive theoretical background for the stress and strength 
criteria implemented in the post-processing. These criteria are fundamental 
for assessing material failure and structural integrity in engineering applications.

* Principal stress analysis
* Equivalent stress measures (von Mises, octahedral)
* Maximum shear stress criterion (Tresca)
* Mohr-Coulomb failure criterion
* Drucker-Prager failure criterion

Each criterion serves specific purposes and is applicable to different material behaviors 
and loading conditions.

For more details, refer to the following wikipedia page: `Yield surface <https://en.wikipedia.org/wiki/Yield_surface>`_.


.. _theory_principal_stresses:

Principal Stresses
==================

Principal stresses are the normal stresses acting on planes where shear stresses vanish. 
They represent the maximum and minimum normal stresses at a point and provide fundamental 
information about the stress state.
At any point in a continuum, there exist three mutually orthogonal principal directions 
along which only normal stresses act. These stresses are denoted as:

.. math::

    p_1 \geq p_2 \geq p_3

where:

* :math:`p_1` = Maximum (major) principal stress
* :math:`p_2` = Intermediate principal stress  
* :math:`p_3` = Minimum (minor) principal stress

The principal stresses are obtained by solving the characteristic equation:

.. math::

    \det(\sigma_{ij} - p\delta_{ij}) = 0

which yields a cubic equation in terms of the stress invariants.

**Applications**

Principal stresses are essential for:

* Determining maximum normal and shear stresses
* Evaluating failure criteria
* Understanding stress distribution in structures
* Analyzing anisotropic material behavior


.. _theory_maximum_shear_stress:

Maximum Shear Stress (Tresca Criterion)
=========================================

The maximum shear stress criterion, also known as the *Tresca criterion*, is one of 
the earliest and simplest failure theories for ductile materials.

The maximum shear stress occurs on planes oriented at 45° to the principal stress 
directions. According to the Tresca criterion, yielding begins when the maximum 
shear stress reaches a critical value.

.. math::

    \tau_{\max} = \frac{1}{2}(p_1 - p_3)

where :math:`p_1` and :math:`p_3` are the maximum and minimum principal stresses, 
respectively.

**Failure Criterion**

Material failure occurs when:

.. math::

    \tau_{\max} \geq \tau_y

where :math:`\tau_y = \sigma_y / 2` is the shear yield strength, and :math:`\sigma_y` 
is the tensile yield strength.

Equivalently, in terms of principal stresses:

.. math::

    p_1 - p_3 \geq \sigma_y

**Applications**

The Tresca criterion is particularly suitable for:

* Ductile metals under proportional loading
* Conservative design estimates
* Cases where maximum shear stress is of primary concern
* Thin-walled pressure vessels


.. _theory_von_mises:

Von Mises Equivalent Stress
============================

The von Mises stress is a scalar measure of stress intensity that accounts for 
the combined effect of all stress components. It is widely used for predicting 
yielding in ductile materials.
The von Mises criterion is based on the distortion energy theory, which states 
that yielding occurs when the distortion energy per unit volume reaches a critical 
value. This criterion is particularly accurate for ductile materials that yield 
by shear rather than by volume change.

For a general 3D stress state with stress tensor components :math:`\sigma_{ij}`:

.. math::

    \sigma_{VM} = \sqrt{\frac{(\sigma_{11}-\sigma_{22})^2 + (\sigma_{22}-\sigma_{33})^2 + (\sigma_{33}-\sigma_{11})^2}{2} + 3(\sigma_{12}^2 + \sigma_{23}^2 + \sigma_{31}^2)}

In terms of principal stresses:

.. math::

    \sigma_{VM} = \sqrt{\frac{(p_1-p_2)^2 + (p_2-p_3)^2 + (p_3-p_1)^2}{2}}

The von Mises stress can be expressed using the second deviatoric stress invariant :math:`J_2`:

.. math::

    \sigma_{VM} = \sqrt{3J_2}

where:

.. math::

    J_2 = \frac{1}{6}[(p_1-p_2)^2 + (p_2-p_3)^2 + (p_3-p_1)^2]

**Failure Criterion**

Material yielding occurs when:

.. math::

    f = \sigma_{VM} - \sigma_y \geq 0

where :math:`\sigma_y` is the uniaxial yield strength.

**Applications**

The von Mises criterion is widely used for:

* Ductile materials (metals, polymers)
* Complex stress states
* Fatigue analysis
* Finite element analysis
* Multiaxial loading conditions

**Advantages**

* Smooth yield surface (no corners)
* Insensitive to hydrostatic pressure
* Experimentally validated for many ductile materials
* Mathematically convenient for computational analysis


.. _theory_octahedral_stress:

Octahedral Stresses
===================

Octahedral stresses are the normal and shear stress components acting on the 
octahedral plane, which is equally inclined to all three principal stress directions.
The octahedral plane makes equal angles (54.74°) with each principal stress axis. 
The stresses on this plane provide a measure of the mean stress and deviatoric 
stress components.

The *octahedral normal stress* represents the mean or hydrostatic stress:

.. math::

    \sigma_{oct} = \frac{I_1}{3} = \frac{p_1 + p_2 + p_3}{3}

This is identical to the mean stress:

.. math::

    \sigma_m = \frac{\sigma_{11} + \sigma_{22} + \sigma_{33}}{3}

The *octahedral shear stress* measures the intensity of shear:

.. math::

    \tau_{oct} = \sqrt{\frac{2}{3}J_2} = \frac{1}{3}\sqrt{2[(p_1-p_2)^2 + (p_2-p_3)^2 + (p_3-p_1)^2]}

Expanded form in terms of stress tensor components:

.. math::

    \tau_{oct} = \frac{1}{3}\sqrt{(\sigma_{11}-\sigma_{22})^2 + (\sigma_{22}-\sigma_{33})^2 + (\sigma_{33}-\sigma_{11})^2 + 6(\sigma_{12}^2 + \sigma_{23}^2 + \sigma_{31}^2)}

**Relation to Von Mises Stress**

The octahedral shear stress and von Mises stress are closely related:

.. math::

    \sigma_{VM} = \sqrt{\frac{3}{2}} \cdot \sqrt{2} \cdot \tau_{oct} = \sqrt{3}\tau_{oct}

or inversely:

.. math::

    \tau_{oct} = \frac{\sqrt{2}}{\sqrt{3}}\sigma_{VM} = \sqrt{\frac{2}{3}}\sigma_{VM}

**Physical Meaning**

* **Octahedral normal stress** (:math:`\sigma_{oct}`): Represents the hydrostatic 
  component of stress, which causes volumetric deformation without shape change.
  
* **Octahedral shear stress** (:math:`\tau_{oct}`): Represents the deviatoric 
  component of stress, which causes shape change (distortion) without volume change.

**Applications**

Octahedral stresses are particularly useful in:

* Soil mechanics and geotechnical engineering
* Pressure-dependent material models
* Understanding stress decomposition into volumetric and deviatoric parts
* Plasticity theory


.. _theory_mohr_coulomb:

Mohr-Coulomb Failure Criterion
===============================

The Mohr-Coulomb criterion is one of the most widely used failure criteria for 
geomaterials such as soils, rocks, and concrete. It accounts for the effect of 
normal stress on shear strength.

The Mohr-Coulomb criterion states that failure occurs on a plane when the shear 
stress on that plane reaches a value that depends linearly on the normal stress. 
This reflects the physical observation that most geomaterials are stronger under 
compression (confinement) than under tension.

The criterion can be expressed in two equivalent forms:

1. **c-φ form**: Using cohesion and friction angle
2. **Tension-Compression form**: Using uniaxial strengths


.. _theory_mohr_coulomb_c_phi:

Mohr-Coulomb: c-φ Form
----------------------

This is the classical form used in soil mechanics and rock mechanics.
The failure envelope in the τ-σ plane (shear stress vs. normal stress) is:

.. math::

    \tau = c + \sigma \tan\phi

where:

* :math:`c` = Cohesion (shear strength at zero normal stress)
* :math:`\phi` = Internal friction angle (angle of shearing resistance)
* :math:`\sigma` = Normal stress on the failure plane
* :math:`\tau` = Shear stress on the failure plane

**Equivalent Stress Formulation**

For each principal stress pair :math:`(p_i, p_j)`, the failure function is:

.. math::

    \tau_{ij} &= \frac{1}{2}|p_i - p_j| \\
    \sigma_{ij} &= \frac{1}{2}(p_i + p_j) \\
    f_{ij} &= \frac{\tau_{ij}}{\cos\phi} - \sigma_{ij}\tan\phi

The equivalent stress is the maximum among all three pairs:

.. math::

    \sigma_{eq} = \max(f_{12}, f_{13}, f_{23})

**Failure Criterion**

Failure occurs when:

.. math::

    f = \sigma_{eq} - c \geq 0

Alternatively, in the standard form:

.. math::

    \frac{\tau}{\cos\phi} - \sigma\tan\phi = c

**Parameters**

* **Cohesion (c)**: Represents the shear strength when no normal stress is present. 
  For cohesionless materials (sands), c = 0.
  
* **Friction angle (φ)**: Typically ranges from 25° to 45° for soils. Higher values 
  indicate greater frictional resistance.

**Applications**

* Soil mechanics and foundation design
* Rock mechanics and slope stability
* Concrete and masonry structures
* Earth pressure calculations
* Bearing capacity analysis


.. _theory_mohr_coulomb_sy:

Mohr-Coulomb: Tension-Compression Form
---------------------------------------

This form is convenient when uniaxial test data is available.
Define the strength ratio:

.. math::

    m = \frac{s_{yc}}{s_{yt}}

where:

* :math:`s_{yc}` = Uniaxial compression strength (positive)
* :math:`s_{yt}` = Uniaxial tension strength (positive)

Define the coefficient:

.. math::

    K = \frac{m-1}{m+1}

For each principal stress pair:

.. math::

    t_{ij} = |p_i - p_j| + K(p_i + p_j)

The equivalent stress is:

.. math::

    \sigma_{eq} = \frac{1}{2}(m+1)\max(t_{12}, t_{13}, t_{23})

**Failure Criterion**

.. math::

    f = \sigma_{eq} - s_{yc} \geq 0

**Relation Between Forms**

The two forms are related by:

.. math::

    c &= \frac{2s_{yc}s_{yt}}{s_{yc}+s_{yt}} \cos\phi \\
    \sin\phi &= \frac{s_{yc}-s_{yt}}{s_{yc}+s_{yt}} \\
    m &= \frac{1+\sin\phi}{1-\sin\phi}

**Special Cases**

* **φ = 0**: Reduces to Tresca criterion (cohesive material with no friction)
* **c = 0**: Pure frictional material (cohesionless soil like sand)
* **m = 1** (:math:`s_{yc} = s_{yt}`): Symmetric yielding (like metals)


.. _theory_drucker_prager:

Drucker-Prager Failure Criterion
=================================

The Drucker-Prager criterion is a smooth approximation to the Mohr-Coulomb criterion. 
It is widely used in finite element analysis due to its mathematical convenience and 
smooth yield surface.

The Drucker-Prager criterion extends the von Mises criterion to include the effect 
of hydrostatic pressure. It is particularly suitable for pressure-dependent materials 
such as soils, rocks, concrete, and polymers.

The criterion can be expressed in two forms:

1. **c-φ form**: Matched to Mohr-Coulomb parameters
2. **Tension-Compression form**: Using uniaxial strengths


.. _theory_drucker_prager_c_phi:

Drucker-Prager: c-φ Form
-------------------------

The general Drucker-Prager failure surface is:

.. math::

    f = \sqrt{J_2} - BI_1 - A = 0

where:

* :math:`I_1 = p_1 + p_2 + p_3` = First stress invariant (mean stress × 3)
* :math:`J_2 = \frac{1}{6}[(p_1-p_2)^2 + (p_2-p_3)^2 + (p_3-p_1)^2]` = Second deviatoric stress invariant
* :math:`A`, :math:`B` = Material parameters related to cohesion and friction

**Matching to Mohr-Coulomb**

Since the Drucker-Prager surface is circular in the deviatoric plane while Mohr-Coulomb 
is hexagonal, different matching schemes are used:

**Type 1: Circumscribed (Outer Bound)**

The Drucker-Prager cone circumscribes the Mohr-Coulomb hexagon, touching at the 
compression meridian:

.. math::

    A &= \frac{6c\cos\phi}{\sqrt{3}(3-\sin\phi)} \\
    B &= \frac{2\sin\phi}{\sqrt{3}(3-\sin\phi)}

This provides a conservative (safe) estimate.

**Type 2: Middle (Average)**

The cone passes through the vertices and edges of the hexagon:

.. math::

    A &= \frac{6c\cos\phi}{\sqrt{3}(3+\sin\phi)} \\
    B &= \frac{2\sin\phi}{\sqrt{3}(3+\sin\phi)}

This is often called the "plane strain" matching.

**Type 3: Inscribed (Inner Bound)**

The cone is inscribed in the hexagon, touching at the tension and compression meridians:

.. math::

    A &= \frac{3c\cos\phi}{\sqrt{9+3\sin^2\phi}} \\
    B &= \frac{\sin\phi}{\sqrt{9+3\sin^2\phi}}

This provides the least conservative estimate.

**Equivalent Stress and Strength**

.. math::

    \sigma_{eq} &= \sqrt{J_2} - BI_1 \\
    \sigma_y &= A

**Failure Criterion**

.. math::

    f = \sigma_{eq} - A \geq 0

**Parameters**

* **A**: Related to cohesion, determines the size of the yield surface
* **B**: Related to friction angle, determines the pressure sensitivity


.. _theory_drucker_prager_sy:

Drucker-Prager: Tension-Compression Form
-----------------------------------------

Define the strength ratio:

.. math::

    m = \frac{s_{yc}}{s_{yt}}

Calculate the generalized shear stress:

.. math::

    q = \sqrt{\frac{1}{2}[(p_1-p_2)^2 + (p_2-p_3)^2 + (p_3-p_1)^2]}

Note that :math:`q = \sqrt{3J_2}` relates to the octahedral shear stress.

**Equivalent Stress**

.. math::

    \sigma_{eq} = \frac{1}{2}(m-1)I_1 + \frac{1}{2}(m+1)q

**Failure Criterion**

.. math::

    f = \sigma_{eq} - s_{yc} \geq 0

where :math:`s_{yc}` is the uniaxial compression strength.

**Relation to Standard Form**

The parameters relate to the standard Drucker-Prager form as:

.. math::

    \alpha = B &= \frac{m-1}{2\sqrt{3}} \\
    k = A &= \frac{m+1}{2\sqrt{3}}s_{yc}

**Special Cases**

* **m = 1** (:math:`s_{yc} = s_{yt}`): Reduces to von Mises criterion (pressure-independent)
* **B = 0**: Reduces to von Mises criterion
* **Large B**: Material is highly pressure-sensitive


Comparison of Criteria
=======================

+-------------------+--------------------------------+--------------------------------+
| Criterion         | Advantages                     | Limitations                    |
+===================+================================+================================+
| **Tresca**        | - Simple                       | - Overly conservative          |
|                   | - Conservative                 | - Ignores intermediate stress  |
|                   | - Clear physical meaning       | - Sharp corners                |
+-------------------+--------------------------------+--------------------------------+
| **Von Mises**     | - Smooth surface               | - Pressure-insensitive         |
|                   | - Widely validated             | - Not for brittle materials    |
|                   | - Computationally efficient    | - Predicts equal T/C strength  |
+-------------------+--------------------------------+--------------------------------+
| **Mohr-Coulomb**  | - Captures pressure effect     | - Hexagonal surface (corners)  |
|                   | - Well-established for soils   | - Singularities at apex        |
|                   | - Physical parameters          | - Computational challenges     |
+-------------------+--------------------------------+--------------------------------+
| **Drucker-Prager**| - Smooth surface               | - Approximate matching         |
|                   | - Pressure-dependent           | - Less physical parameters     |
|                   | - Computationally convenient   | - Multiple matching options    |
+-------------------+--------------------------------+--------------------------------+

Material Selection Guide
------------------------

* **Ductile metals**: von Mises, Tresca
* **Soils and rocks**: Mohr-Coulomb (c-φ), Drucker-Prager
* **Concrete**: Mohr-Coulomb, Drucker-Prager
* **Polymers**: von Mises (low pressure), Drucker-Prager (high pressure)
* **Ceramics**: Maximum principal stress (brittle failure)
