import numpy as np

from ._plot_resp_base import PlotResponseBase


class PlotUnstruResponseBase(PlotResponseBase):
    def __init__(self, model_info_steps, resp_step, model_update, nodal_resp_steps=None):
        super().__init__(model_info_steps, resp_step, model_update, nodal_resp_steps)
        self.ele_type = "Shell"

    def _check_input(self):
        if self.ele_type.lower() == "shell":
            self.ele_type = "Shell"
            self.resp_type, self.component, self.fiber_point = _check_input_shell(
                self.resp_type, self.component, fiber_pts=self.fiber_point
            )
        elif self.ele_type.lower() == "plane":
            self.ele_type = "Plane"
            self.resp_type, self.component = _check_input_plane(self.resp_type, self.component)
        elif self.ele_type.lower() in ["brick", "solid"]:
            self.ele_type = "Brick"
            self.resp_type, self.component = _check_input_solid(self.resp_type, self.component)
        else:
            raise ValueError(f"Not supported element type {self.ele_type}! Valid options are: Shell, Plane, Brick.")  # noqa: TRY003

    def _get_unstru_da(self, step):
        if self.ele_type.lower() == "shell":
            return self._get_model_da("ShellData", step)
        elif self.ele_type.lower() == "plane":
            return self._get_model_da("PlaneData", step)
        elif self.ele_type.lower() in ["brick", "solid"]:
            return self._get_model_da("BrickData", step)
        else:
            raise ValueError(f"Invalid element type {self.ele_type}! Valid options are: Shell, Plane, Brick.")  # noqa: TRY003

    def _set_comp_resp_type(self, ele_type, resp_type, component, fiber_point=None):
        self.ele_type = ele_type
        self.resp_type = resp_type
        self.component = component
        self.fiber_point = fiber_point  # for shell elements only

        self._check_input()

    def _make_unstru_info(self, ele_tags, step):
        pos = self._get_node_da(step)
        unstru_data = self._get_unstru_da(step)
        if ele_tags is None:
            tags, cell_types, cells = self._get_unstru_cells(unstru_data)
        else:
            tags = np.atleast_1d(ele_tags)
            cells = unstru_data.sel(eleTags=tags)
            tags, cell_types, cells = self._get_unstru_cells(cells)
        return tags, pos, cells, cell_types

    def refactor_resp_step(self, ele_tags, ele_type, resp_type: str, component: str, fiber_point=None):
        self._set_comp_resp_type(ele_type, resp_type, component, fiber_point=fiber_point)
        resps = []

        for i in range(self.num_steps):
            da = self._get_resp_da(i, self.resp_type, self.component)

            if self.ModelUpdate or ele_tags is not None:
                tags, pos, _, _ = self._make_unstru_info(ele_tags, i)
                if "eleTags" in da.dims:
                    da = da.sel(eleTags=tags)
            else:
                pos = self._get_node_da(i)

            resps.append(self._process_scalar_from_da(da, pos, fiber_point))

        self.resp_step = resps

    def _process_scalar_from_da(self, da, pos, fiber_point):
        """process the response data array to extract scalar values."""

        def _reset_fiber_point(fiber_point, da):
            if fiber_point == "top":
                fiber_point = da.coords["fiberPoints"].values[-1]
            elif fiber_point == "bottom":
                fiber_point = da.coords["fiberPoints"].values[0]
            elif fiber_point == "middle":
                fiber_point = da.coords["fiberPoints"].values[len(da.coords["fiberPoints"]) // 2]
            return fiber_point

        if "nodeTags" in da.dims:  # response at nodes
            scalars = pos.sel(coords="x").copy() * 0
            if "fiberPoints" in da.dims:  # response at fiber points (shells)
                fiber_point = _reset_fiber_point(fiber_point, da)
                da = da.sel(fiberPoints=fiber_point)
            scalars.loc[{"nodeTags": da.coords["nodeTags"]}] = da
            return scalars

        if "fiberPoints" in da.dims and "GaussPoints" in da.dims:  # response at fiber points and Gauss points (shells)
            fiber_point = _reset_fiber_point(fiber_point, da)
            da = da.sel(fiberPoints=fiber_point)
            return da.sel(fiberPoints=fiber_point).mean(dim="GaussPoints", skipna=True)

        if "GaussPoints" in da.dims:  # response at Gauss points
            return da.mean(dim="GaussPoints", skipna=True)

        return da  # fallback: return raw data

    def _get_resp_peak(self, idx="absMax"):
        if isinstance(idx, str):
            if idx.lower() == "absmax":
                resp = [np.max(np.abs(data)) for data in self.resp_step]
                step = np.argmax(resp)
            elif idx.lower() == "max":
                resp = [np.max(data) for data in self.resp_step]
                step = np.argmax(resp)
            elif idx.lower() == "absmin":
                resp = [np.min(np.abs(data)) for data in self.resp_step]
                step = np.argmin(resp)
            elif idx.lower() == "min":
                resp = [np.min(data) for data in self.resp_step]
                step = np.argmin(resp)
            else:
                raise ValueError("Invalid argument, one of [absMax, absMin, Max, Min]")  # noqa: TRY003
        else:
            step = int(idx)
        cmin, cmax = self._get_resp_clim()
        return step, (cmin, cmax)

    def _get_resp_clim(self):
        maxv = [np.max(data) for data in self.resp_step]
        minv = [np.min(data) for data in self.resp_step]
        cmin, cmax = np.min(minv), np.max(maxv)
        self.clim = (cmin, cmax)
        return cmin, cmax

    def _get_mesh_data(self, *args, **kwargs):
        pass

    def _make_title(self, *args, **kwargs):
        pass

    def _create_mesh(self, *args, **kwargs):
        pass

    def _update_mesh(self, *args, **kwargs):
        pass

    def plot_slide(self, *args, **kwargs):
        pass

    def plot_peak_step(self, *args, **kwargs):
        pass

    def plot_anim(self, *args, **kwargs):
        pass


def _check_input_shell(resp_type, resp_dof, fiber_pts=None):
    if resp_type is None:
        resp_type = "sectionForces"
    resp_type_lower = resp_type.lower()

    valid_resp_map = {
        "sectionforces": "sectionForces",
        "sectiondeformations": "sectionDeformations",
        "sectionforcesatnodes": "sectionForcesAtNodes",
        "sectiondeformationsatnodes": "sectionDeformationsAtNodes",
        "stresses": "Stresses",
        "strains": "Strains",
        "stressesatnodes": "StressesAtNodes",
        "strainsatnodes": "StrainsAtNodes",
    }

    if resp_type_lower not in valid_resp_map:
        raise ValueError(
            f"Not supported GP response type {resp_type}! Valid options are: "
            + ", ".join(valid_resp_map.values())
            + "."
        )

    resp_type = valid_resp_map[resp_type_lower]

    if "section" in resp_type.lower():
        valid_dofs = {"fxx", "fyy", "fxy", "mxx", "myy", "mxy", "vxz", "vyz"}
        if resp_dof is None:
            resp_dof = "MXX"
    else:
        valid_dofs = {"sigma11", "sigma22", "sigma12", "sigma23", "sigma13"}
        if resp_dof is None:
            resp_dof = "sigma11"
        if fiber_pts is None:
            fiber_pts = "top"
        elif isinstance(fiber_pts, str):
            fiber_pts = fiber_pts.lower()
            if fiber_pts not in {"top", "bottom", "middle"}:
                raise ValueError(f"Not supported fiber points {fiber_pts}! Valid options are: top, bottom, middle.")  # noqa: TRY003
        else:
            fiber_pts = int(fiber_pts)

    if resp_dof.lower() not in valid_dofs:
        raise ValueError(  # noqa: TRY003
            f"Not supported component {resp_dof}! Valid options are: {', '.join(d.upper() for d in valid_dofs)}."
        )

    return resp_type, resp_dof, fiber_pts


def _check_input_plane(resp_type, resp_dof):
    if resp_type is None:
        resp_type = "Stresses"

    resp_type_lower = resp_type.lower()
    type_map = {
        "stresses": "Stresses",
        "stress": "Stresses",
        "stressesatnodes": "StressesAtNodes",
        "stressatnodes": "StressesAtNodes",
        "strains": "Strains",
        "strain": "Strains",
        "strainsatnodes": "StrainsAtNodes",
        "strainatnodes": "StrainsAtNodes",
    }

    if resp_type_lower not in type_map:
        raise ValueError(  # noqa: TRY003
            f"Not supported response type {resp_type}! "
            "Valid options are: Stresses, StressesAtNodes, Strains, StrainsAtNodes"
        )

    is_stress = "stress" in resp_type_lower
    is_node = "nodes" in resp_type_lower

    if resp_dof is None:
        resp_dof = "sigma_vm"

    resp_dof_lower = resp_dof.lower()
    tensor_dofs = {"sigma11", "sigma22", "sigma12"}
    measure_dofs = {"p1", "p2", "sigma_vm", "tau_max"}

    if resp_dof_lower in measure_dofs:
        resp_type = ("StressMeasures" if is_stress else "StrainMeasures") + ("AtNodes" if is_node else "")
    elif resp_dof_lower in tensor_dofs:
        resp_type = ("Stresses" if is_stress else "Strains") + ("AtNodes" if is_node else "")
        if not is_stress:
            resp_dof = resp_dof_lower.replace("sigma", "eps")
    else:
        raise ValueError(  # noqa: TRY003
            f"Not supported component {resp_dof}! "
            "Valid options are: sigma11, sigma22, sigma12, p1, p2, sigma_vm, tau_max."
        )

    return resp_type, resp_dof


def _check_input_solid(resp_type, resp_dof):
    if resp_type is None:
        resp_type = "Stresses"

    resp_type_lower = resp_type.lower()
    type_map = {
        "stresses": "Stresses",
        "stress": "Stresses",
        "stressesatnodes": "StressesAtNodes",
        "stressatnodes": "StressesAtNodes",
        "strains": "Strains",
        "strain": "Strains",
        "strainsatnodes": "StrainsAtNodes",
        "strainatnodes": "StrainsAtNodes",
    }

    if resp_type_lower not in type_map:
        raise ValueError(  # noqa: TRY003
            f"Not supported response type {resp_type}! "
            "Valid options are: Stresses, StressesAtNodes, Strains, StrainsAtNodes"
        )

    is_stress = "stress" in resp_type_lower
    is_node = "nodes" in resp_type_lower

    if resp_dof is None:
        resp_dof = "sigma_vm"

    resp_dof_lower = resp_dof.lower()
    tensor_dofs = {"sigma11", "sigma22", "sigma33", "sigma12", "sigma23", "sigma13"}
    measure_dofs = {"p1", "p2", "p3", "sigma_vm", "tau_max", "sigma_oct", "tau_oct"}

    if resp_dof_lower in measure_dofs:
        resp_type = ("StressMeasures" if is_stress else "StrainMeasures") + ("AtNodes" if is_node else "")
    elif resp_dof_lower in tensor_dofs:
        resp_type = ("Stresses" if is_stress else "Strains") + ("AtNodes" if is_node else "")
        if not is_stress:
            resp_dof = resp_dof_lower.replace("sigma", "eps")
    else:
        raise ValueError(  # noqa: TRY003
            f"Not supported component {resp_dof}! "
            "Valid options are: sigma11, sigma22, sigma33, sigma12, sigma23, sigma13, "
            "p1, p2, p3, sigma_vm, tau_max, sigma_oct, tau_oct."
        )

    return resp_type, resp_dof
