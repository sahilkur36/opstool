import numpy as np
import xarray as xr
import openseespy.opensees as ops

from ._response_base import ResponseBase, _expand_to_uniform_array


# from ._response_extrapolation import resp_extrap_tri3, resp_extrap_tri6
# from ._response_extrapolation import (
#     resp_extrap_quad4,
#     resp_extrap_quad8,
#     resp_extrap_quad9,
# )


class PlaneRespStepData(ResponseBase):

    def __init__(self, ele_tags=None):
        self.resp_names = [
            "Stresses",
            "Strains",
        ]
        self.resp_steps = None
        self.step_track = 0
        self.ele_tags = ele_tags
        self.times = []
        self.initialize()

    def initialize(self):
        self.resp_steps = []
        self.add_data_one_step(self.ele_tags)
        self.times = [0.0]
        self.step_track = 0

    def reset(self):
        self.initialize()

    def add_data_one_step(self, ele_tags):
        stresses, strains = _get_gauss_resp(ele_tags)
        data_vars = dict()
        data_vars["Stresses"] = (["eleTags", "GaussPoints", "DOFs"], stresses)
        data_vars["Strains"] = (["eleTags", "GaussPoints", "DOFs"], strains)
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "eleTags": ele_tags,
                "GaussPoints": np.arange(strains.shape[1])+1,
                "DOFs": ["sigma11", "sigma22", "sigma12", "p1", "p2", "sigma_vm", "tau_max"],
            },
            attrs={
                "sigma11, sigma22, sigma12": "Normal stress and shear stress (strain) in the x-y plane.",
                "p1, p2": "Principal stresses (strains).",
                "sigma_vm": "Von Mises stress.",
                "tau_max": "Maximum shear stress (strains).",
            },
        )
        self.resp_steps.append(ds)
        self.times.append(ops.getTime())
        self.step_track += 1

    def _to_xarray(self):
        self.resp_steps = xr.concat(self.resp_steps, dim="time", join="outer")
        self.resp_steps.coords["time"] = self.times

    def get_data(self):
        return self.resp_steps

    def get_track(self):
        return self.step_track

    def save_file(self, dt: xr.DataTree):
        self._to_xarray()
        dt["/PlaneResponses"] = self.resp_steps
        return dt

    @staticmethod
    def read_file(dt: xr.DataTree):
        resp_steps = dt["/PlaneResponses"].to_dataset()
        return resp_steps

    @staticmethod
    def read_response(dt: xr.DataTree, resp_type: str = None, ele_tags=None):
        ds = PlaneRespStepData.read_file(dt)
        if resp_type is None:
            if ele_tags is None:
                return ds
            else:
                return ds.sel(eleTags=ele_tags)
        else:
            if resp_type not in list(ds.keys()):
                raise ValueError(
                    f"resp_type {resp_type} not found in {list(ds.keys())}"
                )
            if ele_tags is not None:
                return ds[resp_type].sel(eleTags=ele_tags)
            else:
                return ds[resp_type]


def _get_gauss_resp(ele_tags):
    stresses, strains = [], []
    for etag in ele_tags:
        etag = int(etag)
        stress = ops.eleResponse(etag, "stresses")
        strain = ops.eleResponse(etag, "strains")
        stresses.append(np.reshape(stress, (-1, 3)))
        strains.append(np.reshape(strain, (-1, 3)))
    stresses = _expand_to_uniform_array(stresses)
    strains = _expand_to_uniform_array(strains)
    stress_measures = _calculate_stresses_measures(stresses)
    strain_measures = _calculate_stresses_measures(strains)
    stresses = np.concatenate((stresses, stress_measures), axis=-1)
    strains = np.concatenate((strains, strain_measures), axis=-1)
    return stresses, strains


def _calculate_stresses_measures(stress_array):
    """
    Calculate various stresses from the stress values at Gaussian points.

    Parameters:
    stress_array (numpy.ndarray): A 3D array with shape (num_elements, num_gauss_points, num_stresses).

    Returns:
    dict: A dictionary containing the calculated stresses for each element.
    """
    # Extract individual stress components
    sig11 = stress_array[..., 0]  # Normal stress in x-direction
    sig22 = stress_array[..., 1]  # Normal stress in y-direction
    sig12 = stress_array[..., 2]  # Normal stress in z-direction

    # Calculate von Mises stress for each Gauss point
    sig_vm = np.sqrt(sig11 ** 2 - sig11 * sig22 + sig22 ** 2 + 3 * sig12 ** 2)

    # Calculate principal stresses (eigenvalues)
    p1 = (sig11 + sig22) / 2 + np.sqrt(((sig11 - sig22) / 2) ** 2 + sig12 ** 2)
    p2 = (sig11 + sig22) / 2 - np.sqrt(((sig11 - sig22) / 2) ** 2 + sig12 ** 2)

    # Calculate maximum shear stress
    tau_max = np.sqrt(((sig11 - sig22) / 2) ** 2 + sig12 ** 2)

    data = np.stack([p1, p2, sig_vm, tau_max], axis=0)

    return data

# ----------------------------------------------------------------------------------------------
#
#
# def _get_plane_resp(ele_tags, node_tags):
#     all_nodal_stress, all_nodal_strain = dict(), dict()
#     for ntag in node_tags:
#         all_nodal_stress[ntag] = []
#         all_nodal_strain[ntag] = []
#     for etag in ele_tags:
#         ntags = ops.eleNodes(etag)
#         if len(ntags) == 3:
#             nodal_stress, nodal_strain = _get_resp_tri3(etag)
#         elif len(ntags) == 6:
#             nodal_stress, nodal_strain = _get_resp_tri6(etag)
#         elif len(ntags) == 4:
#             nodal_stress, nodal_strain = _get_resp_quad4(etag)
#         elif len(ntags) == 8:
#             nodal_stress, nodal_strain = _get_resp_quad8(etag)
#         elif len(ntags) == 9:
#             nodal_stress, nodal_strain = _get_resp_quad9(etag)
#         else:
#             raise RuntimeError("Unsupported planar element type!")
#         for i, ntag in enumerate(ntags):
#             all_nodal_stress[ntag].append(nodal_stress[i])
#             all_nodal_strain[ntag].append(nodal_strain[i])
#     return all_nodal_stress, all_nodal_strain
#
#
# def _get_resp_tri3(etag):
#     stress = ops.eleResponse(etag, "integrPoint", "1", "stress")
#     stress = [stress[0], stress[1], 0.0, stress[2], 0.0, 0.0]
#     strain = ops.eleResponse(etag, "integrPoint", "1", "strain")
#     strain = [strain[0], strain[1], 0.0, strain[2], 0.0, 0.0]
#     stress, strain = _get_all_resp(stress, strain)
#     nodal_stress = resp_extrap_tri3(stress)
#     nodal_strain = resp_extrap_tri3(strain)
#     return nodal_stress, nodal_strain
#
#
# def _get_resp_tri6(etag):
#     stress, strain = [], []
#     for i in range(3):
#         stressi = ops.eleResponse(etag, "integrPoint", f"{i + 1}", "stress")
#         stressi = [stressi[0], stressi[1], 0.0, stressi[2], 0.0, 0.0]
#         straini = ops.eleResponse(etag, "integrPoint", f"{i + 1}", "strain")
#         straini = [straini[0], straini[1], 0.0, straini[2], 0.0, 0.0]
#         stressi, straini = _get_all_resp(stressi, straini)
#         stress.append(stressi)
#         strain.append(straini)
#     nodal_stress = resp_extrap_tri6(stress)
#     nodal_strain = resp_extrap_tri6(strain)
#     return nodal_stress, nodal_strain
#
#
# def _get_resp_quad4(etag):
#     stress, strain = [], []
#     for i in range(4):
#         stressi = ops.eleResponse(etag, "integrPoint", f"{i + 1}", "stress")
#         stressi = [stressi[0], stressi[1], 0.0, stressi[2], 0.0, 0.0]
#         straini = ops.eleResponse(etag, "integrPoint", f"{i + 1}", "strain")
#         straini = [straini[0], straini[1], 0.0, straini[2], 0.0, 0.0]
#         stressi, straini = _get_all_resp(stressi, straini)
#         stress.append(stressi)
#         strain.append(straini)
#     nodal_stress = resp_extrap_quad4(stress)
#     nodal_strain = resp_extrap_quad4(strain)
#     return nodal_stress, nodal_strain
#
#
# def _get_resp_quad8(etag):
#     stress, strain = [], []
#     for i in range(9):
#         stressi = ops.eleResponse(etag, "integrPoint", f"{i + 1}", "stress")
#         stressi = [stressi[0], stressi[1], 0.0, stressi[2], 0.0, 0.0]
#         straini = ops.eleResponse(etag, "integrPoint", f"{i + 1}", "strain")
#         straini = [straini[0], straini[1], 0.0, straini[2], 0.0, 0.0]
#         stressi, straini = _get_all_resp(stressi, straini)
#         stress.append(stressi)
#         strain.append(straini)
#     nodal_stress = resp_extrap_quad8(stress)
#     nodal_strain = resp_extrap_quad8(strain)
#     return nodal_stress, nodal_strain
#
#
# def _get_resp_quad9(etag):
#     stress, strain = [], []
#     for i in range(9):
#         stressi = ops.eleResponse(etag, "integrPoint", f"{i + 1}", "stress")
#         stressi = [stressi[0], stressi[1], 0.0, stressi[2], 0.0, 0.0]
#         straini = ops.eleResponse(etag, "integrPoint", f"{i + 1}", "strain")
#         straini = [straini[0], straini[1], 0.0, straini[2], 0.0, 0.0]
#         stressi, straini = _get_all_resp(stressi, straini)
#         stress.append(stressi)
#         strain.append(straini)
#     nodal_stress = resp_extrap_quad9(stress)
#     nodal_strain = resp_extrap_quad9(strain)
#     return nodal_stress, nodal_strain
#
#
# def _get_all_resp(stress, strain):
#     stress2 = _get_principal_resp(stress)
#     stress += stress2
#     strain2 = _get_principal_resp(strain)
#     strain += strain2
#     return stress, strain
#
#
# def _get_principal_resp(resp):
#     resp_mat = np.array(
#         [
#             [resp[0], resp[3], resp[5]],
#             [resp[3], resp[1], resp[4]],
#             [resp[5], resp[4], resp[2]],
#         ]
#     )
#     eigenvalues, _ = np.linalg.eig(resp_mat)
#     principal_values = np.sort(eigenvalues)[::-1]
#     p1, p2, p3 = principal_values
#     tau_max = np.max([(p1 - p2) / 2, (p2 - p3) / 2, (p3 - p1) / 2])
#     sigma_vm = np.sqrt(0.5 * ((p1 - p2) ** 2 - (p2 - p3) ** 2 - (p3 - p1) ** 2))
#     sigma_oct = (p1 + p2 + p3) / 3
#     tau_oct = np.sqrt(((p1 - p2) ** 2 - (p2 - p3) ** 2 - (p3 - p1) ** 2) / 9)
#     return p1, p2, p3, tau_max, sigma_vm, sigma_oct, tau_oct